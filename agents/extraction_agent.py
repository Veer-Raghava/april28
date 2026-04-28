"""
agents/extraction_agent.py — Agent 4: Document → Structured Rows.

Key improvements:
  1. TABLE-FIRST: tables are sent to LLM before text chunks
  2. Source URL always comes from the pipeline record, never from LLM output
  3. Live append: each row is written to CSV immediately after extraction
  4. Junk sections (References, Acknowledgements) are stripped before LLM
  5. Stronger extraction prompt: explicit anti-hallucination rules
  6. ADC + domain-aware chunk scoring
  7. Schema-driven field definitions with extraction location hints
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.console_setup import console
from agents.base_agent import BaseAgent
import config as cfg
from tools.llm_client import is_empty, parse_json_response

try:
    from tools.image_chem_extractor import extract_chemical_structures_from_figures
    _IMAGE_CHEM_AVAILABLE = True
except ImportError:
    _IMAGE_CHEM_AVAILABLE = False

# Junk section headers — chunks starting with these are excluded
_JUNK_HEADERS = {
    "references", "bibliography", "acknowledgment", "acknowledgement",
    "funding", "conflict of interest", "author contribution",
    "supplementary", "abbreviation", "ethical approval",
    "data availability", "supporting information",
}

# Domain-relevant scoring keywords (science + ADC-specific)
_DOMAIN_KEYWORDS = {
    # ADC-specific
    "dar", "adc", "antibody", "payload", "linker", "conjugate",
    "mmae", "dm1", "dxd", "maytansine", "auristatin", "her2", "cd30",
    "trop-2", "trop2", "ic50", "ec50", "cleavable", "noncleavable",
    "drug-to-antibody", "val-cit", "mcc", "spdb", "maleimide",
    # General primary data markers
    "table", "figure", "we measured", "we synthesized", "we prepared",
    "n=", "result", "assay", "conjugation efficiency",
    "synthesized", "characterised", "characterized",
}


class ExtractionAgent(BaseAgent):

    def __init__(self, config: dict):
        super().__init__(agent_id="extraction_agent", config=config)
        self._output_path: str = str(config.get("output_path", ""))

    def _load_schema(self) -> dict:
        schema_path = self.state_dir / "schema.json"
        if not schema_path.exists():
            return {}
        try:
            with schema_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    async def run(self) -> None:
        console.print("\n[bold cyan]🔬 Agent 4: Extraction[/bold cyan] — starting")

        schema         = self._load_schema()
        vision_columns = schema.get("requires_vision", [])
        schema_defs    = schema.get("field_definitions", {})

        doc_records = self.read_input_queue("documents.jsonl")
        pending     = [r for r in doc_records if r.get("status") == "pending_extraction"]

        if not pending:
            console.print("  [dim yellow]No pending_extraction documents[/dim yellow]")
            self.log_status("run", "skip", {"reason": "no pending documents"})
            return

        total_rows = 0
        all_cols_seen: list[str] = []

        for doc in pending:
            source      = doc.get("source", "")
            topic       = doc.get("topic", "")
            columns     = [c for c in doc.get("columns", [])
                           if c not in ("Source_URL", "Dataset_Links")]

            if not columns:
                console.print(f"  [yellow]⚠ No columns for {source[:60]}[/yellow]")
                continue
            all_cols_seen = columns  # for live CSV header

            console.print(f"\n  [bold]📄 Extracting:[/bold] {source[:80]}")
            self.log_status("started", "success", {"source": source})

            field_defs  = schema_defs if schema_defs else doc.get("field_definitions", {})
            field_exs   = doc.get("field_examples", {})
            merge_cfg   = doc.get("merge_config", {})
            full_text   = doc.get("full_text", "")
            text_blocks = doc.get("text_blocks", [])
            tables      = doc.get("tables", [])
            figures     = doc.get("figures", [])

            field_block  = self._build_field_block(columns, topic, field_defs)
            example_json = self._build_example(columns, topic, field_exs)

            all_rows: list[dict] = []

            # ── Pass 1: TABLE-FIRST (most data-dense) ────────────────────
            console.print(f"  [dim]📊 Tables: {len(tables)} found[/dim]")
            for i, tbl in enumerate(tables[:8]):
                console.print(f"  [dim]  Table {i+1}/{min(len(tables),8)}…[/dim]")
                header   = tbl.get("header", [])
                caption  = tbl.get("caption", "")
                hdr_line = ("Headers: " + ", ".join(header) + "\n") if header else ""
                tbl_text = (f"Caption: {caption}\n{hdr_line}\n{tbl['csv']}"
                            if caption else f"{hdr_line}{tbl['csv']}")
                tbl_rows = self._extract_from_text(
                    tbl_text, columns, field_block, example_json, topic,
                    is_table=True,
                )
                all_rows.extend(tbl_rows)
                console.print(f"  [dim]    → {len(tbl_rows)} row(s)[/dim]")

            # ── Pass 2: Text chunks ───────────────────────────────────────
            console.print(f"  [dim]📝 Text: {len(full_text):,} chars[/dim]")
            text_context = self._add_metadata_context(doc)
            text_rows    = self._extract_from_text(
                text_context, columns, field_block, example_json, topic,
                is_table=False, chunks=text_blocks,
            )
            all_rows.extend(text_rows)
            console.print(f"  [dim]  → {len(text_rows)} row(s) from text[/dim]")

            # ── Pass 3: Figure captions ───────────────────────────────────
            for fig in figures[:5]:
                if fig.get("caption") and len(fig["caption"]) > 20:
                    fig_rows = self._extract_from_text(
                        f"Figure caption: {fig['caption']}",
                        columns, field_block, example_json, topic,
                        is_table=False,
                    )
                    all_rows.extend(fig_rows)

            # ── Pass 4: Vision (chemical structures) ─────────────────────
            if vision_columns and cfg.ENABLE_VISION and _IMAGE_CHEM_AVAILABLE:
                vision_rows = self._extract_vision_fields(
                    figures, vision_columns, source, topic)
                if vision_rows:
                    all_rows.extend(vision_rows)
                    console.print(f"  [dim]Vision → {len(vision_rows)} row(s)[/dim]")

            # ── Merge, dedup, yield check ─────────────────────────────────
            merged = self._merge_rows(all_rows, columns, merge_cfg)

            yield_rate = self._compute_yield(merged, columns)
            if yield_rate < cfg.MIN_YIELD_RATE and len(text_blocks) > 1:
                console.print(
                    f"  [yellow]⚠ Low yield ({yield_rate:.0%}) — retrying…[/yellow]")
                missing_cols = self._find_weak_columns(merged, columns)
                retry_rows   = self._retry_extraction(
                    text_blocks, columns, missing_cols,
                    field_block, example_json, topic,
                )
                if retry_rows:
                    all_rows.extend(retry_rows)
                    merged = self._merge_rows(all_rows, columns, merge_cfg)

            # ── Write rows — LIVE append to both JSONL and CSV ───────────
            for row in merged:
                # Strip any LLM-hallucinated URL fields
                row.pop("Source_URL",  None)
                row.pop("source_url",  None)
                row.pop("Source URL",  None)

                confidence = {
                    col: ("high" if not is_empty(row.get(col, "")) else "missing")
                    for col in columns
                }
                record = {
                    "data":           row,
                    "source_url":     source,   # ALWAYS from ingestion record
                    "confidence":     confidence,
                    "row_confidence": self._score_row(row, columns),
                    "issues":         [],
                    "status":         "pending_validation",
                    "ts":             time.time(),
                    "columns":        columns,
                    "topic":          topic,
                }
                self.write_output("extracted_rows.jsonl", record)

                # Live CSV append
                self._live_append_csv(row, source, columns)

            total_rows += len(merged)
            self.log_status("completed", "success",
                            {"source": source, "rows": len(merged),
                             "yield": f"{self._compute_yield(merged, columns):.0%}"})
            console.print(f"  [green]✓ {len(merged)} row(s) extracted & written[/green]")

        console.print(
            f"\n[bold cyan]🔬 Extraction Agent[/bold cyan] — "
            f"done. {total_rows} total rows."
        )

    def _live_append_csv(self, row: dict, source_url: str,
                          columns: list[str]) -> None:
        """Append a single row immediately to the output CSV."""
        if not self._output_path:
            return
        try:
            path    = Path(self._output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            is_new  = not path.exists() or path.stat().st_size == 0
            all_cols = columns + ["Source_URL", "row_confidence"]

            with open(path, "a", newline="", encoding="utf-8-sig") as fh:
                writer = csv.DictWriter(fh, fieldnames=all_cols,
                                        extrasaction="ignore")
                if is_new:
                    writer.writeheader()
                flat = dict(row)
                flat["Source_URL"]     = source_url
                flat["row_confidence"] = round(self._score_row(row, columns), 2)
                # Fill missing columns
                for col in all_cols:
                    if col not in flat:
                        flat[col] = "N/A"
                writer.writerow(flat)
        except Exception as e:
            console.print(f"  [dim yellow]Live CSV append failed: {e}[/dim yellow]")

    # ------------------------------------------------------------------ #
    #  Prompt building                                                     #
    # ------------------------------------------------------------------ #

    def _build_field_block(self, columns: list[str], topic: str,
                            defs: dict[str, str]) -> str:
        lines = []
        for col in columns:
            key  = col.strip()
            defn = next(
                (v for k, v in defs.items()
                 if k in key.lower() or key.lower() in k),
                f'Extract the value for "{key}" from the text about {topic}. '
                'Look in tables, results sections, and methods.'
            )
            lines.append(f'  "{key}": {defn}')
        return "\n".join(lines)

    def _build_example(self, columns: list[str], topic: str,
                        examples: dict) -> str:
        if examples:
            example = {}
            for col in columns[:10]:
                key = col.strip()
                val = next(
                    (v for k, v in examples.items()
                     if k.lower() in key.lower() or key.lower() in k.lower()),
                    "actual value from text"
                )
                example[key] = val
            return json.dumps([example], indent=2)
        return json.dumps([{col: "extracted value" for col in columns[:10]}], indent=2)

    def _extraction_system(self, field_block: str, example_json: str,
                            topic: str, is_table: bool) -> str:
        entity_rule = (
            "This text is a DATA TABLE — return ONE JSON object per data row. "
            "Preserve all data rows. Do NOT merge rows."
            if is_table else
            "If the text describes multiple distinct entities, return one object "
            "per entity. If it discusses a single entity, return one object."
        )
        return (
            f'You are a scientific data extraction specialist. Topic: "{topic}".\n\n'
            "TASK: Extract structured data from the provided text and return ONLY "
            "a valid JSON array.\n\n"
            f"FIELD DEFINITIONS (with extraction hints):\n{field_block}\n\n"
            f"EXAMPLE OUTPUT (format only — DO NOT copy these values):\n{example_json}\n\n"
            "ABSOLUTE RULES — violating any is a critical error:\n"
            "1. Return ONLY a JSON array — no prose, no markdown fences.\n"
            '2. Use exactly the field names shown. Use "N/A" when absent.\n'
            "3. NEVER copy field definition text as a value.\n"
            "4. NEVER use your own knowledge — extract ONLY from the provided text.\n"
            "5. NEVER hallucinate values. N/A is always correct when data is absent.\n"
            "6. NEVER include Source_URL, source, or URL fields — these are added separately.\n"
            "7. Values must be concise and verbatim from the source text (< 20 words each).\n"
            f"{entity_rule}"
        )

    # ------------------------------------------------------------------ #
    #  Chunk scoring                                                       #
    # ------------------------------------------------------------------ #

    def _structural_score(self, chunk: str, idx: int, total: int) -> float:
        cl = chunk.lower()[:500]
        if any(junk in cl for junk in _JUNK_HEADERS):
            return -1.0
        score = 0.0
        if sum(1 for s in ['|', '\t', '---', ',"'] if s in chunk) >= 2:
            score += 0.25
        numbers = len(re.findall(r'\b\d+(?:\.\d+)?\b', chunk.lower()))
        if numbers > 5:
            score += min(numbers * 0.012, 0.20)
        kw_hits = sum(1 for kw in _DOMAIN_KEYWORDS if kw in cl)
        score += min(kw_hits * 0.05, 0.30)
        if idx == 0:
            score += 0.10
        if len(chunk.strip()) < 150:
            score -= 0.3
        return max(min(score, 1.0), 0.0)

    def _select_chunks(self, chunks: list[str], columns: list[str],
                        max_chunks: int) -> list[str]:
        if len(chunks) <= max_chunks:
            return chunks
        scores = [self._structural_score(c, i, len(chunks))
                  for i, c in enumerate(chunks)]
        # Build a domain-aware query embedding
        col_str = " ".join(columns)
        dom_kws = " ".join(list(_DOMAIN_KEYWORDS)[:15])
        query   = f"Extract data for: {col_str}. Data types: {dom_kws}"
        embeddings = self.llm.get_embeddings([query] + chunks)
        if embeddings:
            query_emb  = embeddings[0]
            chunk_embs = embeddings[1:]
            sem_scores = [self._cosine_sim(query_emb, ce) for ce in chunk_embs]
            hybrid = [
                -1.0 if scores[i] == -1.0
                else 0.60 * sem_scores[i] + 0.40 * scores[i]
                for i in range(len(chunks))
            ]
        else:
            hybrid = scores
        valid = [(i, s) for i, s in enumerate(hybrid) if s > 0]
        valid.sort(key=lambda x: -x[1])
        selected = set(v[0] for v in valid[:max_chunks])
        selected.add(0)
        # Add adjacent chunks for context
        for i in list(selected):
            for n in [i - 1, i + 1]:
                if 0 <= n < len(chunks) and n not in selected and hybrid[n] > 0:
                    selected.add(n)
        if len(selected) > max_chunks:
            ranked   = sorted(selected, key=lambda i: hybrid[i], reverse=True)
            selected = set(ranked[:max_chunks])
        return [chunks[i] for i in sorted(selected)]

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na  = sum(x * x for x in a) ** 0.5
        nb  = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    # ------------------------------------------------------------------ #
    #  Extraction methods                                                  #
    # ------------------------------------------------------------------ #

    def _extract_from_text(self, text: str, columns: list[str],
                            field_block: str, example_json: str,
                            topic: str, is_table: bool = False,
                            chunks: list[str] | None = None) -> list[dict]:
        system = self._extraction_system(field_block, example_json, topic, is_table)
        if self.llm.provider in ("claude", "openai"):
            if len(text) <= 180_000:
                raw = self.llm.complete(text, system=system, max_tokens=8192)
                return self._parse_and_normalize(raw, columns)
            else:
                from tools.pdf_tools import chunk_text
                big_chunks = chunk_text(text, max_chars=180_000, overlap=1000)
                all_rows: list[dict] = []
                for i, ch in enumerate(big_chunks):
                    console.print(f"    [dim]API chunk {i+1}/{len(big_chunks)}…[/dim]")
                    raw = self.llm.complete(ch, system=system, max_tokens=8192)
                    all_rows.extend(self._parse_and_normalize(raw, columns))
                return all_rows
        else:  # Ollama
            if chunks is None:
                from tools.pdf_tools import chunk_text
                chunks = chunk_text(text, max_chars=cfg.MAX_TEXT_CHARS * 2,
                                    overlap=300)
            selected = self._select_chunks(chunks, columns, cfg.OLLAMA_MAX_CHUNKS)
            console.print(f"    [dim]Selected {len(selected)}/{len(chunks)} chunks[/dim]")
            all_rows = []
            for i, chunk in enumerate(selected):
                prompt = (
                    f"FIELD DEFINITIONS:\n{field_block}\n\n"
                    f"EXAMPLE OUTPUT:\n{example_json}\n\n"
                    f"TEXT:\n---\n{chunk}\n---\n\n"
                    "Return JSON array only:"
                )
                try:
                    raw  = self.llm.complete(prompt, system=system,
                                              max_tokens=1024)
                    rows = self._parse_and_normalize(raw, columns)
                    all_rows.extend(rows)
                    console.print(f"    [dim]Chunk {i+1} → {len(rows)} row(s)[/dim]")
                except Exception as e:
                    console.print(f"    [yellow]⚠ chunk {i+1} failed: {e}[/yellow]")
            return all_rows

    def _parse_and_normalize(self, raw: str, columns: list[str]) -> list[dict]:
        parsed = parse_json_response(raw)
        if not parsed:
            return []
        results = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            norm = {k.strip().lower(): str(v).strip() for k, v in item.items()}
            row  = {}
            for col in columns:
                key = col.strip().lower()
                val = norm.get(key)
                if val is None:
                    for k, v in norm.items():
                        if key in k or k in key:
                            val = v; break
                # Strip any URL values the LLM may have inserted
                if val and any(x in str(val).lower()
                               for x in ["http://", "https://", "doi.org"]):
                    if "url" in key or "source" in key or "link" in key:
                        val = None
                row[col] = val if val and not is_empty(val) else "N/A"
            results.append(row)
        return results

    def _filter_rows(self, rows: list[dict], columns: list[str]) -> list[dict]:
        min_valid = max(1, int(len(columns) * 0.30))
        good = [r for r in rows
                if sum(1 for c in columns
                       if not is_empty(r.get(c, "N/A"))) >= min_valid]
        return good or rows[:1]

    # ------------------------------------------------------------------ #
    #  Merge & dedup                                                       #
    # ------------------------------------------------------------------ #

    def _merge_rows(self, rows: list[dict], columns: list[str],
                    merge_config: dict) -> list[dict]:
        if not rows:
            return []
        identity_kw = set(merge_config.get("identity", ["name", "title"]))
        content_kw  = set(merge_config.get("content",  ["description", "year"]))
        anchor = None
        for col in columns:
            cl = col.lower()
            if (any(kw in cl for kw in identity_kw) and
                    not any(kw in cl for kw in content_kw)):
                anchor = col; break

        groups: dict[Any, list[dict]] = defaultdict(list)
        if anchor:
            for row in rows:
                key = row.get(anchor, "").strip().lower()
                # Normalize anchor for fuzzy grouping
                key = re.sub(r'\s*[\(\[].*?[\)\]]', '', key).strip()
                groups[key if not is_empty(key) else id(row)].append(row)
        else:
            for idx, row in enumerate(rows):
                groups[idx].append(row)

        merged = []
        for items in groups.values():
            row = {}
            for col in columns:
                cl    = col.lower()
                is_id = any(kw in cl for kw in identity_kw)
                vals  = [it.get(col, "N/A") for it in items
                         if not is_empty(it.get(col, "N/A"))]
                if not vals:
                    row[col] = "N/A"
                elif is_id:
                    row[col] = max(vals, key=len)
                else:
                    unique, seen_set = [], set()
                    for v in vals:
                        vl = v.strip().lower()
                        if vl not in seen_set:
                            seen_set.add(vl); unique.append(v.strip())
                    row[col] = "; ".join(unique[:12])
            merged.append(row)

        seen_sigs: set[str] = set()
        deduped = []
        for row in merged:
            sig = "|".join(str(row.get(c, "")).lower() for c in columns)
            if sig not in seen_sigs:
                seen_sigs.add(sig); deduped.append(row)
        return self._filter_rows(deduped, columns)

    # ------------------------------------------------------------------ #
    #  Yield / retry                                                       #
    # ------------------------------------------------------------------ #

    def _compute_yield(self, rows: list[dict], columns: list[str]) -> float:
        if not rows:
            return 0.0
        total  = len(rows) * len(columns)
        filled = sum(1 for r in rows for c in columns
                     if not is_empty(r.get(c, "N/A")))
        return filled / total if total else 0.0

    def _find_weak_columns(self, rows: list[dict], columns: list[str]) -> list[str]:
        return [col for col in columns
                if sum(1 for r in rows if is_empty(r.get(col, "N/A")))
                / max(len(rows), 1) > 0.5]

    def _retry_extraction(self, text_blocks: list[str], columns: list[str],
                           missing_cols: list[str], field_block: str,
                           example_json: str, topic: str) -> list[dict]:
        if not missing_cols:
            return []
        system = (
            f'You are a data extraction expert. Topic: "{topic}".\n'
            f'The PREVIOUS extraction missed these fields: {missing_cols}\n'
            'Look specifically in tables and results sections.\n'
            'Return ONLY a JSON array. Use exact field names.\n'
            f'FIELD DEFINITIONS:\n{field_block}'
        )
        max_chunks = min(cfg.OLLAMA_MAX_CHUNKS + 2, len(text_blocks))
        selected   = self._select_chunks(text_blocks, missing_cols, max_chunks)
        all_rows: list[dict] = []
        for chunk in selected[:3]:
            raw = self.llm.complete(chunk, system=system, max_tokens=1024)
            all_rows.extend(self._parse_and_normalize(raw, columns))
        return all_rows

    # ------------------------------------------------------------------ #
    #  Vision extraction                                                   #
    # ------------------------------------------------------------------ #

    def _extract_vision_fields(self, figures: list[dict],
                                vision_columns: list[str],
                                source: str, topic: str = "") -> list[dict]:
        if not figures or not vision_columns or not cfg.ENABLE_VISION:
            return []
        if not _IMAGE_CHEM_AVAILABLE:
            return []
        try:
            return extract_chemical_structures_from_figures(
                figures=figures,
                source_url=source,
                vision_columns=vision_columns,
                topic=topic,
            )
        except Exception as e:
            console.print(f"  [yellow]⚠ Vision extraction error: {e}[/yellow]")
            return []

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _score_row(self, row: dict, columns: list[str]) -> float:
        score = 1.0
        for col in columns:
            if is_empty(row.get(col, "N/A")):
                score -= 0.1
        return max(0.0, min(1.0, score))

    def _add_metadata_context(self, doc: dict) -> str:
        extra = []
        da = doc.get("data_availability", "")
        if da:
            extra.append(f"[DATA AVAILABILITY]\n{da}")
        dlinks = doc.get("dataset_links", [])
        if dlinks:
            extra.append("[DATASET LINKS]\n" +
                         "; ".join(f"{d['type']}: {d['value']}"
                                   for d in dlinks[:5]))
        suplinks = doc.get("supplement_links", [])
        if suplinks:
            extra.append("[SUPPLEMENTARY FILES]\n" +
                         ", ".join(s["filename"] for s in suplinks[:5]))
        extra_text = "\n\n".join(extra)
        return (doc.get("full_text", "") +
                ("\n\n" + extra_text if extra_text else ""))
