"""
agents/null_hunter.py — Agent 5: Systematic N/A Filler.

Five recovery passes, in order:
  Pass 1 — Stable field carry-forward (title, authors, year, identity cols)
  Pass 2 — Multi-API science backfill:
            • CrossRef (DOI → bibliographic metadata)
            • PubChem  (drug/compound name → SMILES, formula, CAS, MW)
            • PubMed   (PMID from URL → abstract → LLM extraction)
            • UniProt  (protein/target name → function, gene, organism)
            • ChEMBL   (compound name → bioactivity data)
  Pass 3 — Context inference (LLM reasons from known fields in same row
            to guess the likely value — for universal/fixed scientific facts)
  Pass 4 — Targeted re-extraction (focused LLM prompt against document tables
            + text blocks for high-null columns)
  Pass 5 — Confidence update + row re-scoring

The most common ADC N/A pattern: SMILES / formula for a well-known payload
like MMAE or DM1. These are fixed facts — PubChem returns them instantly.
Context inference handles cases like "if ADC Name is Kadcyla → Target = HER2".
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests as _requests

from tools.console_setup import console
from agents.base_agent import BaseAgent
from tools.llm_client import is_empty
import config as cfg


# Stable fields that should be the same across all rows from one source
STABLE_FIELDS = {
    "title", "authors", "author", "journal", "publication", "source",
    "year", "date", "doi", "publisher", "institution", "affiliation",
    "organism", "species",
}


class NullHunterAgent(BaseAgent):
    """
    Agent 5 — Systematically fills N/A values using 5 recovery passes.
    """

    def __init__(self, config: dict):
        super().__init__(agent_id="null_hunter", config=config)

    async def run(self) -> None:
        console.print("\n[bold cyan]🔎 Agent 5: Null Hunter[/bold cyan] — starting")

        rows = self.read_input_queue("extracted_rows.jsonl")
        if not rows:
            console.print("  [dim yellow]No rows in extracted_rows.jsonl[/dim yellow]")
            self.log_status("run", "skip", {"reason": "no extracted rows"})
            return

        columns = next((r.get("columns", []) for r in rows if r.get("columns")), [])
        columns = [c for c in columns if c not in ("Source_URL", "Dataset_Links")]
        topic   = next((r.get("topic", "") for r in rows if r.get("topic")), "")

        # Load schema for null strategy
        schema = self._load_schema()
        null_strategy = schema.get("null_strategy", {})
        identity_cols = schema.get("identity_columns", [])

        console.print(f"  [dim]{len(rows)} rows | {len(columns)} columns | topic: {topic}[/dim]")
        self.log_status("started", "success", {"rows": len(rows), "topic": topic})

        initial_nulls = self._count_nulls(rows, columns)
        console.print(f"  [dim]Initial N/A count: {initial_nulls}[/dim]")

        # ── Pass 1: Stable carry-forward ──────────────────────────────────
        console.print("  [cyan]Pass 1:[/cyan] Stable field carry-forward…")
        self._pass1_carry_forward(rows, columns, identity_cols)

        # ── Pass 2: Multi-API science backfill ────────────────────────────
        console.print("  [cyan]Pass 2:[/cyan] Science API backfill (CrossRef, PubChem, PubMed, UniProt, ChEMBL)…")
        self._pass2_science_apis(rows, columns, topic)

        # ── Pass 3: Context inference ──────────────────────────────────────
        console.print("  [cyan]Pass 3:[/cyan] Context inference (LLM reasons from known fields)…")
        self._pass3_context_inference(rows, columns, topic, schema)

        # ── Pass 4: Targeted re-extraction ────────────────────────────────
        console.print("  [cyan]Pass 4:[/cyan] Targeted LLM re-extraction from documents…")
        self._pass4_targeted_reextract(rows, columns, topic)

        # ── Pass 5: Confidence update ─────────────────────────────────────
        self._pass5_update_confidence(rows, columns)

        final_nulls = self._count_nulls(rows, columns)
        filled = initial_nulls - final_nulls
        console.print(
            f"  [green]✓ Null Hunter: filled {filled} values "
            f"({initial_nulls} → {final_nulls} N/A)[/green]"
        )
        self.log_status("completed", "success",
                        {"filled": filled, "initial": initial_nulls,
                         "final": final_nulls})

        # Overwrite extracted_rows.jsonl with enriched rows
        output_path = self.state_dir / "extracted_rows.jsonl"
        output_path.write_text("", encoding="utf-8")
        for row in rows:
            row["status"] = "pending_validation"
            self.write_output("extracted_rows.jsonl", row)

        console.print(
            f"\n[bold cyan]🔎 Null Hunter[/bold cyan] — "
            f"done. {len(rows)} rows written back."
        )

    # ------------------------------------------------------------------ #
    #  Pass 1: Stable carry-forward                                        #
    # ------------------------------------------------------------------ #

    def _pass1_carry_forward(self, rows: list[dict], columns: list[str],
                              identity_cols: list[str]) -> None:
        filled = 0
        by_source: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_source[row.get("source_url", "")].append(row)

        # Stable = STABLE_FIELDS + schema identity columns
        stable_keywords = STABLE_FIELDS | {c.lower() for c in identity_cols}
        stable_cols = [c for c in columns
                       if any(sf in c.lower() for sf in stable_keywords)]

        for source, source_rows in by_source.items():
            for col in stable_cols:
                best = max(
                    (row.get("data", {}).get(col, "N/A") for row in source_rows),
                    key=lambda v: 0 if is_empty(v) else len(str(v)),
                    default=""
                )
                if best and not is_empty(best):
                    for row in source_rows:
                        if is_empty(row.get("data", {}).get(col, "N/A")):
                            row["data"][col]          = best
                            row["confidence"][col]     = "high"
                            filled += 1
        console.print(f"    [dim]Pass 1: carried forward {filled} values[/dim]")

    # ------------------------------------------------------------------ #
    #  Pass 2: Science API backfill                                        #
    # ------------------------------------------------------------------ #

    def _pass2_science_apis(self, rows: list[dict], columns: list[str],
                             topic: str) -> None:
        filled = 0
        by_source: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_source[row.get("source_url", "")].append(row)

        for source, source_rows in by_source.items():
            # CrossRef via DOI
            doi = self._find_doi(source_rows, columns)
            if doi:
                meta = self._query_crossref(doi)
                if meta:
                    filled += self._apply_crossref(source_rows, columns, meta)

            # PubMed via PMID in source URL
            pmid = self._extract_pmid(source)
            if pmid:
                abstract = self._query_pubmed_abstract(pmid)
                if abstract:
                    filled += self._apply_pubmed(source_rows, columns, abstract, topic)

            # PubChem — for any compound/drug/payload column that is null
            filled += self._apply_pubchem(source_rows, columns)

            # UniProt — for any target/protein column
            filled += self._apply_uniprot(source_rows, columns)

            # ChEMBL — for any compound name
            filled += self._apply_chembl(source_rows, columns)

        console.print(f"    [dim]Pass 2: API backfill filled {filled} values[/dim]")

    def _find_doi(self, source_rows: list[dict], columns: list[str]) -> str | None:
        for row in source_rows:
            for col in columns:
                if "doi" in col.lower():
                    val = row.get("data", {}).get(col, "")
                    if not is_empty(val) and val.startswith("10."):
                        return val
        return None

    def _extract_pmid(self, url: str) -> str | None:
        m = re.search(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', url)
        if m:
            return m.group(1)
        m2 = re.search(r'/(\d{7,9})(?:/|$)', url)
        if m2 and "pubmed" in url.lower():
            return m2.group(1)
        return None

    def _query_crossref(self, doi: str) -> dict | None:
        try:
            url  = f"{cfg.CROSSREF_API_URL}/{doi}"
            resp = _requests.get(
                url,
                headers={"User-Agent": f"DatasetBuilder/1.0 (mailto:{cfg.NCBI_EMAIL})"},
                timeout=10,
            )
            if resp.status_code != 200:
                return None
            data  = resp.json().get("message", {})
            title = (data.get("title", [""])[0]
                     if isinstance(data.get("title"), list)
                     else data.get("title", ""))
            authors_list = []
            for a in data.get("author", [])[:10]:
                n = f"{a.get('given','')} {a.get('family','')}".strip()
                if n:
                    authors_list.append(n)
            year = ""
            for key in ("published-print", "published-online", "created"):
                parts = data.get(key, {}).get("date-parts", [[]])
                if parts and parts[0]:
                    year = str(parts[0][0]); break
            abstract = re.sub(r"<[^>]+>", "",
                               data.get("abstract", ""))[:600]
            journal = (data.get("container-title", [""])[0]
                       if data.get("container-title") else "")
            return {
                "title": title, "authors": ", ".join(authors_list),
                "year": year, "journal": journal,
                "publisher": data.get("publisher", ""),
                "abstract": abstract, "doi": doi,
            }
        except Exception:
            return None

    def _apply_crossref(self, source_rows: list[dict], columns: list[str],
                         meta: dict) -> int:
        filled = 0
        field_map = {
            "title":       meta.get("title", ""),
            "authors":     meta.get("authors", ""),
            "author":      meta.get("authors", ""),
            "year":        meta.get("year", ""),
            "date":        meta.get("year", ""),
            "journal":     meta.get("journal", ""),
            "publication": meta.get("journal", ""),
            "publisher":   meta.get("publisher", ""),
            "doi":         meta.get("doi", ""),
            "abstract":    meta.get("abstract", ""),
        }
        for row in source_rows:
            for col in columns:
                if not is_empty(row.get("data", {}).get(col, "N/A")):
                    continue
                cl = col.lower()
                for fname, fval in field_map.items():
                    if fname in cl and fval:
                        row["data"][col]      = str(fval)
                        row["confidence"][col] = "high"
                        filled += 1
                        break
        return filled

    def _query_pubmed_abstract(self, pmid: str) -> str:
        try:
            url = (f"{cfg.PUBMED_API_URL}/efetch.fcgi?"
                   f"db=pubmed&id={pmid}&rettype=abstract&retmode=text"
                   f"&email={cfg.NCBI_EMAIL}")
            resp = _requests.get(url, timeout=12)
            if resp.status_code == 200:
                return resp.text[:3000]
        except Exception:
            pass
        return ""

    def _apply_pubmed(self, source_rows: list[dict], columns: list[str],
                       abstract: str, topic: str) -> int:
        if not abstract or not source_rows:
            return 0
        # Find high-null columns for this source group
        target_cols = [
            col for col in columns
            if is_empty(source_rows[0].get("data", {}).get(col, "N/A"))
        ]
        if not target_cols:
            return 0
        system = (
            f'You are a scientific data extractor. Topic: "{topic}".\n'
            f'Extract ONLY these fields from the provided abstract: {target_cols}\n'
            'Return ONLY a JSON object with these keys. Use "N/A" if absent.\n'
            'DO NOT invent values. Extract only what is explicitly stated.'
        )
        parsed = self.llm.complete_json(abstract, system=system, max_tokens=512)
        if not isinstance(parsed, dict):
            return 0
        norm = {k.strip().lower(): str(v).strip() for k, v in parsed.items()}
        filled = 0
        for row in source_rows:
            for col in target_cols:
                if is_empty(row.get("data", {}).get(col, "N/A")):
                    key = col.strip().lower()
                    val = norm.get(key) or next(
                        (v for k, v in norm.items() if key in k or k in key), None)
                    if val and not is_empty(val):
                        row["data"][col]      = val
                        row["confidence"][col] = "medium"
                        filled += 1
        return filled

    def _apply_pubchem(self, source_rows: list[dict], columns: list[str]) -> int:
        """
        For compound/drug/payload/linker columns: query PubChem by name.
        PubChem is perfect for universal facts like SMILES, formula, CAS, MW
        which never appear in the paper but are fixed for known compounds.
        """
        filled = 0
        compound_col_kws = {"drug", "compound", "payload", "linker", "molecule",
                            "smiles", "formula", "cas", "mw", "weight", "name",
                            "chemical", "ingredient", "substance"}
        output_col_kws   = {"smiles", "formula", "cas", "mw", "weight",
                            "inchi", "molecular"}

        # Find a "name" column we can use as the lookup key
        name_col = next(
            (c for c in columns
             if any(kw in c.lower() for kw in {"drug", "compound", "payload",
                                                "name", "molecule"})),
            None
        )
        # Find output columns that PubChem can fill
        fill_cols = [c for c in columns
                     if any(kw in c.lower() for kw in output_col_kws)]
        if not name_col or not fill_cols:
            return 0

        for row in source_rows:
            compound_name = row.get("data", {}).get(name_col, "")
            if is_empty(compound_name):
                continue
            # Only query if at least one fill column is null
            if not any(is_empty(row.get("data", {}).get(c, "N/A"))
                       for c in fill_cols):
                continue

            pc_data = self._query_pubchem(compound_name)
            if not pc_data:
                continue

            for col in fill_cols:
                if not is_empty(row.get("data", {}).get(col, "N/A")):
                    continue
                cl = col.lower()
                val = None
                if "smiles" in cl:   val = pc_data.get("canonical_smiles")
                elif "formula" in cl: val = pc_data.get("molecular_formula")
                elif "cas" in cl:     val = pc_data.get("cas_number")
                elif "mw" in cl or "weight" in cl:
                    val = pc_data.get("molecular_weight")
                elif "inchi" in cl:   val = pc_data.get("inchi")
                if val and not is_empty(str(val)):
                    row["data"][col]      = str(val)
                    row["confidence"][col] = "high"
                    filled += 1
        return filled

    def _query_pubchem(self, name: str) -> dict | None:
        """Query PubChem REST API by compound name."""
        try:
            # URL-encode the name
            from urllib.parse import quote
            enc = quote(name.strip())
            url = (f"{cfg.PUBCHEM_API_URL}/compound/name/{enc}/JSON"
                   f"?record_type=2d")
            resp = _requests.get(url, timeout=10)
            if resp.status_code != 200:
                return None
            data  = resp.json()
            props = data.get("PC_Compounds", [{}])[0].get("props", [])
            result: dict = {}
            for prop in props:
                urn   = prop.get("urn", {})
                label = urn.get("label", "").lower()
                name_ = urn.get("name", "").lower()
                val   = prop.get("value", {})
                sval  = val.get("sval", val.get("fval", val.get("ival", "")))
                if "canonical smiles" in label or ("smiles" in label and "canonical" in name_):
                    result["canonical_smiles"] = str(sval)
                elif "molecular formula" in label:
                    result["molecular_formula"] = str(sval)
                elif "molecular weight" in label:
                    result["molecular_weight"] = str(sval)
                elif "inchikey" in label:
                    result["inchikey"] = str(sval)
                elif "inchi" in label:
                    result["inchi"] = str(sval)
            # CAS via synonyms endpoint
            try:
                syn_url = (f"{cfg.PUBCHEM_API_URL}/compound/name/{enc}"
                           f"/synonyms/JSON")
                syn_resp = _requests.get(syn_url, timeout=8)
                if syn_resp.status_code == 200:
                    syns = (syn_resp.json().get("InformationList", {})
                            .get("Information", [{}])[0]
                            .get("Synonym", []))
                    cas = next((s for s in syns if re.match(r'^\d{2,7}-\d{2}-\d$', s)),
                               None)
                    if cas:
                        result["cas_number"] = cas
            except Exception:
                pass
            return result if result else None
        except Exception:
            return None

    def _apply_uniprot(self, source_rows: list[dict], columns: list[str]) -> int:
        """Query UniProt for protein/target name → gene, organism, function."""
        filled = 0
        target_name_kws = {"target", "protein", "antigen", "receptor",
                           "enzyme", "biomarker"}
        target_fill_kws = {"gene", "organism", "function", "pathway",
                           "uniprot", "accession"}

        target_col = next(
            (c for c in columns
             if any(kw in c.lower() for kw in target_name_kws)), None
        )
        fill_cols = [c for c in columns
                     if any(kw in c.lower() for kw in target_fill_kws)]
        if not target_col or not fill_cols:
            return 0

        for row in source_rows:
            target_name = row.get("data", {}).get(target_col, "")
            if is_empty(target_name):
                continue
            if not any(is_empty(row.get("data", {}).get(c, "N/A"))
                       for c in fill_cols):
                continue
            up_data = self._query_uniprot(target_name)
            if not up_data:
                continue
            for col in fill_cols:
                if not is_empty(row.get("data", {}).get(col, "N/A")):
                    continue
                cl  = col.lower()
                val = None
                if "gene" in cl:        val = up_data.get("gene")
                elif "organism" in cl:  val = up_data.get("organism")
                elif "function" in cl:  val = up_data.get("function")
                elif "uniprot" in cl or "accession" in cl:
                    val = up_data.get("accession")
                if val and not is_empty(str(val)):
                    row["data"][col]      = str(val)
                    row["confidence"][col] = "high"
                    filled += 1
        return filled

    def _query_uniprot(self, name: str) -> dict | None:
        try:
            from urllib.parse import quote
            url = (f"{cfg.UNIPROT_API_URL}/search?query={quote(name)}"
                   f"&format=json&size=1&fields=accession,gene_names,"
                   f"organism_name,cc_function")
            resp = _requests.get(
                url,
                headers={"Accept": "application/json"},
                timeout=10,
            )
            if resp.status_code != 200:
                return None
            results = resp.json().get("results", [])
            if not results:
                return None
            entry = results[0]
            gene = (entry.get("genes", [{}])[0].get("geneName", {})
                    .get("value", "") if entry.get("genes") else "")
            org  = (entry.get("organism", {}).get("scientificName", ""))
            func_list = entry.get("comments", [])
            func_text = ""
            for c in func_list:
                if c.get("commentType") == "FUNCTION":
                    texts = c.get("texts", [{}])
                    func_text = texts[0].get("value", "")[:300] if texts else ""
                    break
            return {
                "accession": entry.get("primaryAccession", ""),
                "gene":      gene,
                "organism":  org,
                "function":  func_text,
            }
        except Exception:
            return None

    def _apply_chembl(self, source_rows: list[dict], columns: list[str]) -> int:
        """Query ChEMBL for activity data (IC50, Ki, etc.) by compound name."""
        filled = 0
        activity_kws = {"ic50", "ec50", "ki", "kd", "activity",
                        "potency", "inhibition"}
        name_kws     = {"drug", "compound", "payload", "molecule", "name"}

        name_col = next(
            (c for c in columns if any(kw in c.lower() for kw in name_kws)), None)
        fill_cols = [c for c in columns
                     if any(kw in c.lower() for kw in activity_kws)]
        if not name_col or not fill_cols:
            return 0

        for row in source_rows:
            compound = row.get("data", {}).get(name_col, "")
            if is_empty(compound):
                continue
            if not any(is_empty(row.get("data", {}).get(c, "N/A"))
                       for c in fill_cols):
                continue
            chembl_data = self._query_chembl(compound)
            if not chembl_data:
                continue
            for col in fill_cols:
                if not is_empty(row.get("data", {}).get(col, "N/A")):
                    continue
                cl  = col.lower()
                val = None
                for entry in chembl_data:
                    atype = entry.get("standard_type", "").lower()
                    aval  = entry.get("standard_value", "")
                    aunit = entry.get("standard_units", "")
                    if atype and atype in cl and aval:
                        val = f"{aval} {aunit}".strip()
                        break
                if val and not is_empty(val):
                    row["data"][col]      = val
                    row["confidence"][col] = "medium"
                    filled += 1
        return filled

    def _query_chembl(self, name: str) -> list[dict]:
        try:
            from urllib.parse import quote
            # First find ChEMBL ID
            search_url = (f"{cfg.CHEMBL_API_URL}/molecule.json?"
                          f"molecule_synonyms__molecule_synonym__icontains="
                          f"{quote(name)}&limit=1")
            resp = _requests.get(
                search_url, headers={"Accept": "application/json"}, timeout=10)
            if resp.status_code != 200:
                return []
            mols = resp.json().get("molecules", [])
            if not mols:
                return []
            chembl_id = mols[0].get("molecule_chembl_id", "")
            if not chembl_id:
                return []
            # Fetch activities
            act_url = (f"{cfg.CHEMBL_API_URL}/activity.json?"
                       f"molecule_chembl_id={chembl_id}"
                       f"&standard_type__in=IC50,EC50,Ki,Kd"
                       f"&limit=5")
            act_resp = _requests.get(
                act_url, headers={"Accept": "application/json"}, timeout=10)
            if act_resp.status_code != 200:
                return []
            return act_resp.json().get("activities", [])
        except Exception:
            return []

    # ------------------------------------------------------------------ #
    #  Pass 3: Context inference                                           #
    # ------------------------------------------------------------------ #

    def _pass3_context_inference(self, rows: list[dict], columns: list[str],
                                   topic: str, schema: dict) -> None:
        """
        For each row with nulls, give the LLM ALL known values in that row
        and ask it to infer the missing ones based on scientific knowledge.

        This handles universal facts like:
          - "Kadcyla" → Target Antigen = HER2, Antibody = Trastuzumab
          - "MMAE" → Payload class = Auristatin, MOA = Tubulin inhibitor
          - Known DAR values for approved ADCs

        Important: the LLM is explicitly told it may use general scientific
        knowledge for well-known entities, but must mark inferred values
        with confidence "inferred" so they can be validated separately.
        """
        filled = 0
        field_defs = schema.get("field_definitions", {})

        for row in rows:
            data = row.get("data", {})
            null_cols = [c for c in columns if is_empty(data.get(c, "N/A"))]
            if not null_cols:
                continue

            known_vals = {c: data[c] for c in columns
                          if not is_empty(data.get(c, "N/A"))}
            if not known_vals:
                continue  # nothing to infer from

            # Build definitions for null cols
            defs_block = "\n".join(
                f'  "{c}": {field_defs.get(c.lower(), f"The {c} value")}'
                for c in null_cols
            )

            system = (
                f'You are a scientific data expert in "{topic}".\n'
                "You are given a partially-filled row from a research dataset.\n"
                "Based on the KNOWN values in this row, infer the MISSING fields.\n\n"
                "For well-known scientific entities (approved drugs, validated compounds,\n"
                "published ADCs, etc.), you MAY use your general scientific knowledge.\n"
                "For novel/unpublished entities, ONLY infer if you are highly confident.\n\n"
                f"FIELDS TO INFER:\n{defs_block}\n\n"
                'Return ONLY a JSON object with the inferred values.\n'
                'Use "N/A" if you cannot confidently infer a value.\n'
                'DO NOT hallucinate values for unknown entities.'
            )
            prompt = (
                f"KNOWN FIELDS IN THIS ROW:\n"
                + json.dumps(known_vals, indent=2)
                + f"\n\nMISSING FIELDS: {null_cols}\n\nInfer the missing values:"
            )

            parsed = self.llm.complete_json(prompt, system=system, max_tokens=512)
            if not isinstance(parsed, dict):
                continue

            norm = {k.strip().lower(): str(v).strip() for k, v in parsed.items()}
            for col in null_cols:
                if not is_empty(data.get(col, "N/A")):
                    continue
                key = col.strip().lower()
                val = norm.get(key) or next(
                    (v for k, v in norm.items() if key in k or k in key), None)
                if val and not is_empty(val):
                    row["data"][col]      = val
                    row["confidence"][col] = "inferred"
                    filled += 1

        console.print(f"    [dim]Pass 3: inferred {filled} values from context[/dim]")

    # ------------------------------------------------------------------ #
    #  Pass 4: Targeted LLM re-extraction from document                   #
    # ------------------------------------------------------------------ #

    def _pass4_targeted_reextract(self, rows: list[dict], columns: list[str],
                                   topic: str) -> None:
        filled = 0

        col_null_rates = {
            col: sum(1 for r in rows
                     if is_empty(r.get("data", {}).get(col, "N/A")))
                 / max(len(rows), 1)
            for col in columns
        }
        target_cols = [col for col, rate in col_null_rates.items()
                       if rate > 0.4]
        if not target_cols:
            console.print("    [dim]Pass 4: no high-null columns to target[/dim]")
            return

        console.print(f"    [dim]Pass 4: targeting {len(target_cols)} column(s): "
                      f"{target_cols}[/dim]")

        doc_records = self.read_input_queue("documents.jsonl")
        doc_map = {d.get("source", ""): d for d in doc_records}

        by_source: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_source[row.get("source_url", "")].append(row)

        system = (
            f'You are a data extraction expert. Topic: "{topic}".\n'
            f'Find ONLY these missing fields: {target_cols}\n'
            'Search tables first, then text.\n'
            'Return a JSON array of objects. Use "N/A" if genuinely not found.\n'
            'DO NOT invent or hallucinate values.'
        )

        for source, source_rows in by_source.items():
            doc = doc_map.get(source)
            if not doc:
                continue

            # Send tables first (most data-dense)
            chunks_to_try: list[str] = []
            for tbl in doc.get("tables", [])[:6]:
                chunks_to_try.append(tbl.get("csv", ""))
            chunks_to_try.extend(doc.get("text_blocks", [])[:3])

            for chunk in chunks_to_try:
                if not chunk:
                    continue
                parsed = self.llm.complete_json(chunk, system=system,
                                                 max_tokens=512)
                if not parsed:
                    continue
                results = parsed if isinstance(parsed, list) else [parsed]
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    norm = {k.strip().lower(): str(v).strip()
                            for k, v in item.items()}
                    for row in source_rows:
                        for col in target_cols:
                            if not is_empty(row.get("data", {}).get(col, "N/A")):
                                continue
                            key = col.strip().lower()
                            val = norm.get(key) or next(
                                (v for k, v in norm.items()
                                 if key in k or k in key), None)
                            if val and not is_empty(val):
                                row["data"][col]      = val
                                row["confidence"][col] = "low"
                                filled += 1

        console.print(f"    [dim]Pass 4: re-extracted {filled} values[/dim]")

    # ------------------------------------------------------------------ #
    #  Pass 5: Confidence update                                           #
    # ------------------------------------------------------------------ #

    def _pass5_update_confidence(self, rows: list[dict], columns: list[str]) -> None:
        for row in rows:
            score = 1.0
            for col in columns:
                val  = row.get("data", {}).get(col, "N/A")
                conf = row.get("confidence", {}).get(col, "missing")
                if is_empty(val):
                    row["confidence"][col] = "missing"
                    score -= 0.1
                elif conf == "inferred":
                    score -= 0.03    # small penalty for inferred values
                elif conf == "low":
                    score -= 0.05
            row["row_confidence"] = max(0.0, min(1.0, score))

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _load_schema(self) -> dict:
        schema_path = self.state_dir / "schema.json"
        if not schema_path.exists():
            return {}
        try:
            with schema_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _count_nulls(self, rows: list[dict], columns: list[str]) -> int:
        return sum(
            1 for r in rows for c in columns
            if is_empty(r.get("data", {}).get(c, "N/A"))
        )
