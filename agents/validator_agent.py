"""
agents/validator_agent.py — Agent 6: Quality Validation + Assembly.

Agentic Architecture
--------------------
State Bus Contract:
  INPUT  : data/state/extracted_rows.jsonl
           Each line: {"data": {...}, "source_url": str, "confidence": {...},
                       "row_confidence": float, "columns": [...],
                       "status": "pending_validation"}

  OUTPUT : data/state/validated_rows.jsonl  (accepted rows)
           data/state/rejected_rows.jsonl   (low-confidence + blocked rows)
           Final CSV / JSON written to the project output directory.

  LOGS   : data/state/agent_logs.jsonl  (via BaseAgent.log_status)

Multi-tier validation:
  Tier 1: Structural check (schema compliance)
  Tier 2: Format checks (year, DOI, numeric fields)
  Tier 3: Cross-field consistency
  + Confidence scoring → accepted / rejected split → CSV output
"""

from __future__ import annotations

import re
import time
from collections import defaultdict

from tools.console_setup import console
from agents.base_agent import BaseAgent

import config as cfg
from tools.llm_client import is_empty
from tools.export import (
    save_csv, save_rejected, save_json,
    display_preview, display_quality_report,
    display_rejected_rows, display_blocked_sources,
    display_live_row,
)


DOI_REGEX  = re.compile(r"10\.\d{4,}/[^\s,;\"'<>]+")
YEAR_REGEX = re.compile(r"^\d{4}$")


class ValidatorAgent(BaseAgent):
    """
    Agent 6 — Validates, scores, deduplicates, and outputs final data.

    Reads ``extracted_rows.jsonl``, runs three validation tiers, splits rows
    into accepted / rejected, writes to the state bus, and produces the final
    CSV / JSON output files.
    """

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            Must include ``"state_dir"`` and ``"output_path"`` (str, final CSV path).
        """
        super().__init__(agent_id="validator_agent", config=config)
        self.output_path: str = str(config.get("output_path", "data/outputs/output.csv"))

    # ------------------------------------------------------------------ #
    #  Async Run Loop                                                      #
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        """
        Main agent loop.

        1. Reads ``extracted_rows.jsonl``.
        2. Applies Tier 1–3 validation checks.
        3. Re-scores confidence.
        4. Deduplicates globally.
        5. Splits accepted (≥ 0.3) / rejected.
        6. Reads ``triage_blocked.jsonl`` and appends to rejected.
        7. Writes ``validated_rows.jsonl``, ``rejected_rows.jsonl``.
        8. Writes the final CSV and JSON outputs.
        9. Displays rich console reports.
        """
        console.print("\n[bold cyan]✅ Agent 6: Validator + Assembler[/bold cyan] — starting")

        rows = self.read_input_queue("extracted_rows.jsonl")
        if not rows:
            console.print("[red]No data to validate.[/red]")
            self.log_status("run", "skip", {"reason": "no extracted rows"})
            return

        # Derive columns from the first row that carries them
        columns = next(
            (r.get("columns", []) for r in rows if r.get("columns")),
            [],
        )
        columns = [c for c in columns if c not in ("Source_URL", "Dataset_Links")]

        console.print(f"  [dim]{len(rows)} rows | {len(columns)} columns[/dim]")
        self.log_status("started", "success", {"rows": len(rows)})

        issues: dict[str, int] = defaultdict(int)

        # ── Tier 1: Structural check ──────────────────────────────────────
        for row in rows:
            data = row.setdefault("data", {})
            for col in columns:
                if col not in data:
                    data[col] = "N/A"
                    issues["missing_key"] += 1
            # Remove stray keys
            stray = [k for k in list(data.keys())
                     if k not in columns and k not in ("Source_URL", "Dataset_Links")]
            for k in stray:
                del data[k]

        # ── Tier 2: Format checks ─────────────────────────────────────────
        for row in rows:
            data = row.get("data", {})
            row_issues: list[str] = row.setdefault("issues", [])

            for col in columns:
                val = data.get(col, "")
                if is_empty(val):
                    continue

                cl = col.lower()

                # Year: must contain a 4-digit year in 1900–2099 range
                if any(kw in cl for kw in ["year", "date"]):
                    years = re.findall(r'\b(19|20)\d{2}\b', val)
                    if not years:
                        row_issues.append(f"format:{col}:invalid_year")
                        issues["invalid_year"] += 1

                # DOI: must match the standard DOI prefix pattern
                if "doi" in cl:
                    if not DOI_REGEX.match(val):
                        row_issues.append(f"format:{col}:invalid_doi")
                        issues["invalid_doi"] += 1

                # Numeric field: must contain at least one digit
                if any(kw in cl for kw in [
                    "dose", "concentration", "quantity", "amount",
                    "score", "rating", "price", "p-value", "pvalue",
                ]):
                    if not re.search(r'\d', val):
                        row_issues.append(f"format:{col}:no_numeric")
                        issues["no_numeric"] += 1

                # Identity field: must not be a generic placeholder phrase
                if any(kw in cl for kw in ["name", "title", "drug", "compound"]):
                    generic_phrases = [
                        "not specified", "not mentioned", "various",
                        "multiple", "see text", "refer to",
                    ]
                    if val.lower() in generic_phrases or len(val) < 2:
                        row_issues.append(f"format:{col}:generic_value")
                        issues["generic_value"] += 1

        # ── Tier 3: Cross-field consistency ───────────────────────────────
        for row in rows:
            data       = row.get("data", {})
            row_issues = row.get("issues", [])
            source_url = row.get("source_url", "")

            doi_col = next((c for c in columns if "doi" in c.lower()), None)
            if doi_col and source_url:
                doi_val    = data.get(doi_col, "")
                url_doi_m  = DOI_REGEX.search(source_url)
                if doi_val and not is_empty(doi_val) and url_doi_m:
                    url_doi = url_doi_m.group()
                    if doi_val != url_doi and not url_doi.startswith(doi_val):
                        row_issues.append("cross_field:doi_mismatch")
                        issues["doi_mismatch"] += 1

        # ── Confidence re-scoring ─────────────────────────────────────────
        for row in rows:
            score      = row.get("row_confidence", 1.0)
            row_issues = row.get("issues", [])
            confidence = row.get("confidence", {})

            for issue in row_issues:
                if issue.startswith("format:"):
                    score -= 0.15
                elif issue.startswith("cross_field:"):
                    score -= 0.20

            high_conf = sum(1 for v in confidence.values() if v == "high")
            low_conf  = sum(1 for v in confidence.values() if v == "low")
            missing   = sum(1 for v in confidence.values() if v == "missing")

            conf_bonus = (high_conf * 0.02) - (missing * 0.1) - (low_conf * 0.05)
            row["row_confidence"] = max(0.0, min(1.0, score + conf_bonus))

        # ── Global deduplication ──────────────────────────────────────────
        rows = self._global_dedup(rows, columns)

        # ── Split accepted / rejected ─────────────────────────────────────
        accepted: list[dict] = []
        rejected: list[dict] = []

        for row in rows:
            flat = self._to_flat(row, columns)
            if row.get("row_confidence", 0.0) >= 0.3:
                accepted.append(flat)
            else:
                rejected.append(flat)

        # Append blocked sources (from triage) to rejected list
        blocked_records = self.read_input_queue("triage_blocked.jsonl")
        for bs in blocked_records:
            blocked_row = {col: "N/A" for col in columns}
            blocked_row["Source_URL"]     = bs.get("url", "")
            blocked_row["row_confidence"] = 0.0
            blocked_row["_issues"]        = f"blocked:{bs.get('reason', 'unknown')}"
            blocked_row["_block_reason"]  = bs.get("reason", "")
            blocked_row["_block_details"] = bs.get("details", "")
            rejected.append(blocked_row)

        # ── Write state bus outputs ───────────────────────────────────────
        validated_path = self.state_dir / "validated_rows.jsonl"
        validated_path.write_text("", encoding="utf-8")
        for flat in accepted:
            self.write_output("validated_rows.jsonl", flat)

        rejected_bus_path = self.state_dir / "rejected_rows.jsonl"
        rejected_bus_path.write_text("", encoding="utf-8")
        for flat in rejected:
            self.write_output("rejected_rows.jsonl", flat)

        # ── Compute null rates ────────────────────────────────────────────
        null_rates: dict[str, float] = {}
        for col in columns:
            na_count       = sum(1 for r in accepted if is_empty(r.get(col, "N/A")))
            null_rates[col] = na_count / max(len(accepted), 1)

        overall_null     = sum(null_rates.values()) / max(len(null_rates), 1)
        needs_more_sources = overall_null > cfg.MAX_NULL_RATE

        # ── Write final CSV / JSON output files ───────────────────────────
        all_cols   = columns + ["Source_URL", "row_confidence"]
        final_path = save_csv(accepted, all_cols, self.output_path)
        save_json(accepted, self.output_path)

        if rejected:
            import os
            rejected_dir = os.path.dirname(self.output_path) or str(cfg.OUTPUT_DIR)
            save_rejected(rejected, rejected_dir)

        # ── Rich console display ──────────────────────────────────────────
        display_preview(accepted, all_cols)

        display_quality_report(
            accepted=len(accepted),
            rejected=len(rejected),
            null_rates=null_rates,
            issues=dict(issues),
        )

        if rejected:
            display_rejected_rows(rejected, all_cols)

        self.log_status(
            "completed", "success",
            {
                "accepted":     len(accepted),
                "rejected":     len(rejected),
                "blocked":      len(blocked_records),
                "null_rate":    f"{overall_null:.0%}",
                "output_path":  self.output_path,
            },
        )

        if needs_more_sources:
            console.print(
                f"  [yellow]⚠ High null rate ({overall_null:.0%}) "
                "— recommending more sources[/yellow]"
            )

        console.print(
            f"\n[bold cyan]✅ Validator[/bold cyan] — done. "
            f"{len(accepted)} accepted, {len(rejected)} rejected."
        )

        # Return null metrics so the orchestrator can decide on re-search
        return {
            "null_rate":          overall_null,
            "needs_more_sources": needs_more_sources,
            "null_rates":         null_rates,
            "accepted":           len(accepted),
            "rejected":           len(rejected),
        }

    # ------------------------------------------------------------------ #
    #  Deduplication                                                       #
    # ------------------------------------------------------------------ #

    def _global_dedup(self, rows: list[dict], columns: list[str]) -> list[dict]:
        """Remove exact duplicate rows across all sources."""
        seen: set[str] = set()
        deduped        = []
        for row in rows:
            data = row.get("data", {})
            sig  = "|".join(str(data.get(c, "")).lower().strip() for c in columns)
            if sig not in seen:
                seen.add(sig)
                deduped.append(row)

        if len(deduped) < len(rows):
            console.print(f"  [dim]Deduped: {len(rows)} → {len(deduped)} rows[/dim]")

        return deduped

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _to_flat(self, row: dict, columns: list[str]) -> dict:
        """Convert a row record to a flat dict suitable for CSV export."""
        flat                    = dict(row.get("data", {}))
        flat["Source_URL"]      = row.get("source_url", "")
        flat["row_confidence"]  = round(row.get("row_confidence", 0.0), 2)
        row_issues              = row.get("issues", [])
        if row_issues:
            flat["_issues"]     = "; ".join(row_issues)
        return flat
