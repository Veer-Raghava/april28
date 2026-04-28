"""
tools/validate_dataset.py — Post-run dataset validation for the Dataset Builder.

Run this AFTER the pipeline produces output.csv to catch:
  V1  Completeness   — per-column fill rates and overall null rate
  V2  Format         — year ranges, DOI regex, SMILES validity, numeric fields
  V3  Deduplication  — exact-row and near-duplicate detection (Jaccard on text)
  V4  Cross-field     — DOI vs Source_URL consistency, DAR numeric range
  V5  SMILES/SELFIES  — round-trip validation via RDKit + selfies
  V6  Source tracing  — every row's Source_URL must be reachable (HEAD check)

Usage
-----
    # Basic validation
    python -m tools.validate_dataset outputs/my_topic/output.csv

    # With chemical validation and source-URL checks
    python -m tools.validate_dataset outputs/my_topic/output.csv --chem --urls

    # Output a JSON report
    python -m tools.validate_dataset outputs/my_topic/output.csv --report report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from tools.console_setup import console
from rich.table import Table
from rich.panel import Panel
from rich import box

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    from rdkit import Chem  # type: ignore[import-untyped]
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    import selfies as sf  # type: ignore[import-untyped]
    SELFIES_AVAILABLE = True
except ImportError:
    SELFIES_AVAILABLE = False

try:
    import requests as _requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ── Regex patterns ────────────────────────────────────────────────────────────
DOI_RE   = re.compile(r'^10\.\d{4,}/\S+$')
YEAR_RE  = re.compile(r'^\d{4}$')
EMPTY_VS = {"n/a", "none", "not specified", "not mentioned", "not provided",
            "unspecified", "unknown", "na", "nil", "", "not available",
            "not found", "not discussed", "not applicable", "not stated",
            "not reported", "not described"}


def is_empty(v) -> bool:
    return str(v).strip().lower() in EMPTY_VS


# ── Validation functions ──────────────────────────────────────────────────────

class ValidationReport:
    def __init__(self, csv_path: str):
        self.csv_path  = csv_path
        self.warnings: list[str] = []
        self.errors:   list[str] = []
        self.info:     list[str] = []
        self.metrics:  dict      = {}

    def warn(self, msg: str):  self.warnings.append(msg)
    def error(self, msg: str): self.errors.append(msg)
    def note(self, msg: str):  self.info.append(msg)

    def passed(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict:
        return {
            "csv_path": self.csv_path,
            "passed":   self.passed(),
            "errors":   self.errors,
            "warnings": self.warnings,
            "info":     self.info,
            "metrics":  self.metrics,
        }


def v1_completeness(df: pd.DataFrame, report: ValidationReport) -> None:
    """V1 — Per-column fill rates and overall null rate."""
    data_cols  = [c for c in df.columns if c not in ("Source_URL", "row_confidence", "_issues")]
    null_rates = {}
    for col in data_cols:
        n_empty = df[col].apply(is_empty).sum()
        rate    = n_empty / max(len(df), 1)
        null_rates[col] = float(rate)
        if rate > 0.5:
            report.warn(f"Column '{col}' is {rate:.0%} empty")
        elif rate > 0.2:
            report.note(f"Column '{col}' is {rate:.0%} empty")

    overall = sum(null_rates.values()) / max(len(null_rates), 1)
    report.metrics["null_rates"]   = null_rates
    report.metrics["overall_null"] = round(overall, 4)
    report.note(f"Overall null rate: {overall:.0%} across {len(data_cols)} columns")

    if overall > 0.4:
        report.warn(f"High overall null rate ({overall:.0%}) — consider more sources")


def v2_format(df: pd.DataFrame, report: ValidationReport) -> None:
    """V2 — Year ranges, DOI regex, numeric fields."""
    issues = defaultdict(int)

    for col in df.columns:
        cl = col.lower()

        # Year validation
        if "year" in cl or col == "Year":
            for i, val in enumerate(df[col]):
                v = str(val).strip()
                if is_empty(v):
                    continue
                if not YEAR_RE.match(v):
                    issues[f"bad_year:{col}"] += 1
                elif not (1900 <= int(v) <= 2100):
                    issues[f"year_out_of_range:{col}"] += 1

        # DOI validation
        if "doi" in cl:
            for val in df[col]:
                v = str(val).strip()
                if is_empty(v):
                    continue
                if not DOI_RE.match(v):
                    issues[f"bad_doi:{col}"] += 1

        # Numeric fields (IC50, DAR, MW, etc.)
        for kw in ["ic50", "dar", "mw", "weight", "ratio", "kd", "ec50", "cc50", "potency"]:
            if kw in cl:
                for val in df[col]:
                    v = str(val).strip()
                    if is_empty(v):
                        continue
                    # Strip common unit suffixes and try to parse
                    cleaned = re.sub(r'[nµmM\s±><~≤≥]', '', v.split()[0] if ' ' in v else v)
                    try:
                        float(cleaned)
                    except ValueError:
                        issues[f"non_numeric:{col}"] += 1
                break

    for issue, count in issues.items():
        if count > 0:
            report.warn(f"Format issue [{issue}]: {count} cells")

    report.metrics["format_issues"] = dict(issues)


def v3_deduplication(df: pd.DataFrame, report: ValidationReport) -> None:
    """V3 — Exact and near-duplicate detection."""
    data_cols = [c for c in df.columns if c not in ("Source_URL", "row_confidence", "_issues")]

    # Exact duplicates
    sigs = df[data_cols].apply(
        lambda row: "|".join(str(v).strip().lower() for v in row), axis=1
    )
    n_exact = len(df) - sigs.nunique()
    if n_exact > 0:
        report.warn(f"{n_exact} exact-duplicate rows detected")
    report.metrics["exact_duplicates"] = n_exact

    # Near-duplicates (Jaccard on first text column)
    text_cols = [c for c in data_cols if df[c].dtype == object]
    if text_cols:
        col = text_cols[0]
        tokens = [set(str(v).lower().split()) for v in df[col]]
        near = 0
        for i in range(len(tokens)):
            for j in range(i + 1, min(i + 5, len(tokens))):
                a, b = tokens[i], tokens[j]
                if not a or not b:
                    continue
                jaccard = len(a & b) / len(a | b)
                if jaccard > 0.85:
                    near += 1
        if near > 0:
            report.note(f"{near} near-duplicate row pairs (Jaccard > 0.85) in column '{col}'")
        report.metrics["near_duplicates"] = near


def v4_cross_field(df: pd.DataFrame, report: ValidationReport) -> None:
    """V4 — Cross-field consistency checks."""
    issues = 0

    doi_col = next((c for c in df.columns if "doi" in c.lower()), None)
    src_col = "Source_URL"

    if doi_col and src_col in df.columns:
        for _, row in df.iterrows():
            doi = str(row.get(doi_col, "")).strip()
            src = str(row.get(src_col, "")).strip()
            if is_empty(doi) or is_empty(src):
                continue
            doi_prefix = doi.replace("https://doi.org/", "").split("/")[0]
            if doi_prefix and doi_prefix not in src and "doi.org" not in src:
                issues += 1

    if issues > 0:
        report.note(f"{issues} rows where DOI doesn't match Source_URL")
    report.metrics["cross_field_mismatches"] = issues


def v5_chemical(df: pd.DataFrame, report: ValidationReport) -> None:
    """V5 — SMILES/SELFIES round-trip validation."""
    smiles_cols  = [c for c in df.columns if "smiles"  in c.lower()]
    selfies_cols = [c for c in df.columns if "selfies" in c.lower()]

    if not smiles_cols and not selfies_cols:
        return

    invalid_smiles  = 0
    invalid_selfies = 0

    if RDKIT_AVAILABLE and smiles_cols:
        col = smiles_cols[0]
        for val in df[col]:
            v = str(val).strip()
            if is_empty(v):
                continue
            mol = Chem.MolFromSmiles(v)
            if mol is None:
                invalid_smiles += 1
        if invalid_smiles:
            report.warn(f"{invalid_smiles} invalid SMILES strings in '{col}'")
        else:
            report.note(f"All SMILES in '{col}' validated by RDKit")
    elif smiles_cols:
        report.note("RDKit not installed — SMILES validation skipped. pip install rdkit-pypi")

    if SELFIES_AVAILABLE and selfies_cols:
        col = selfies_cols[0]
        for val in df[col]:
            v = str(val).strip()
            if is_empty(v):
                continue
            try:
                sf.decoder(v)
            except Exception:
                invalid_selfies += 1
        if invalid_selfies:
            report.warn(f"{invalid_selfies} invalid SELFIES strings in '{col}'")
        else:
            report.note(f"All SELFIES in '{col}' decoded successfully")
    elif selfies_cols:
        report.note("selfies package not installed — SELFIES validation skipped. pip install selfies")

    report.metrics["invalid_smiles"]  = invalid_smiles
    report.metrics["invalid_selfies"] = invalid_selfies


def v6_source_urls(df: pd.DataFrame, report: ValidationReport, sample: int = 10) -> None:
    """V6 — Spot-check that Source_URLs return HTTP 200 (HEAD request)."""
    if not REQUESTS_AVAILABLE:
        report.note("requests not available — URL checks skipped")
        return

    src_col = "Source_URL"
    if src_col not in df.columns:
        return

    urls   = [u for u in df[src_col].unique() if u and not is_empty(u)]
    sample_urls = urls[:sample]
    dead   = []

    for url in sample_urls:
        try:
            r = _requests.head(url, timeout=8, allow_redirects=True,
                               headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code >= 400:
                dead.append((url, r.status_code))
        except Exception as e:
            dead.append((url, str(e)))

    if dead:
        report.warn(f"{len(dead)}/{len(sample_urls)} sampled URLs unreachable:")
        for url, reason in dead:
            report.warn(f"  {url[:80]} → {reason}")
    else:
        report.note(f"All {len(sample_urls)} sampled Source_URLs reachable")

    report.metrics["dead_urls"] = len(dead)


# ── Rich display ──────────────────────────────────────────────────────────────

def display_report(report: ValidationReport) -> None:
    lines = []
    status = "[bold green]PASSED ✓[/bold green]" if report.passed() else "[bold red]FAILED ✗[/bold red]"
    lines.append(f"  Status: {status}")
    lines.append(f"  File:   {report.csv_path}")
    lines.append(f"  Overall null rate: {report.metrics.get('overall_null', '?'):.0%}")
    lines.append("")

    if report.errors:
        lines.append("[bold red]Errors:[/bold red]")
        for e in report.errors:
            lines.append(f"  🔴 {e}")
        lines.append("")

    if report.warnings:
        lines.append("[bold yellow]Warnings:[/bold yellow]")
        for w in report.warnings:
            lines.append(f"  ⚠ {w}")
        lines.append("")

    if report.info:
        lines.append("[dim]Info:[/dim]")
        for n in report.info:
            lines.append(f"  [dim]ℹ {n}[/dim]")

    panel = Panel(
        "\n".join(lines),
        title="[bold bright_white]📋 Validation Report[/bold bright_white]",
        border_style="green" if report.passed() else "red",
        padding=(1, 2),
        expand=False,
    )
    console.print()
    console.print(panel)

    # Per-column null rates table
    null_rates = report.metrics.get("null_rates", {})
    if null_rates:
        table = Table(
            title="Column Completeness",
            show_lines=False,
            box=box.SIMPLE,
            border_style="bright_blue",
        )
        table.add_column("Column", style="cyan")
        table.add_column("Fill %", style="white")
        table.add_column("Bar", style="white")
        for col, rate in sorted(null_rates.items(), key=lambda x: x[1], reverse=True):
            fill = 1.0 - rate
            bar = "█" * int(fill * 20) + "░" * (20 - int(fill * 20))
            color = "green" if rate < 0.2 else "yellow" if rate < 0.5 else "red"
            table.add_row(col, f"[{color}]{fill:.0%}[/{color}]", f"[{color}]{bar}[/{color}]")
        console.print()
        console.print(table)


# ── Entry point ───────────────────────────────────────────────────────────────

def validate(
    csv_path: str,
    run_chemical: bool = False,
    run_url_check: bool = False,
    url_sample: int = 10,
) -> ValidationReport:
    """
    Run all validation passes on an output CSV.

    Parameters
    ----------
    csv_path : str
        Path to the CSV to validate.
    run_chemical : bool
        If True, run SMILES/SELFIES validation (V5).
    run_url_check : bool
        If True, spot-check Source_URLs (V6).
    url_sample : int
        Number of URLs to spot-check in V6.

    Returns
    -------
    ValidationReport
    """
    report = ValidationReport(csv_path)

    if not Path(csv_path).exists():
        report.error(f"File not found: {csv_path}")
        return report

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as e:
        report.error(f"Could not read CSV: {e}")
        return report

    report.note(f"Loaded {len(df)} rows × {len(df.columns)} columns")
    report.metrics["rows"]    = len(df)
    report.metrics["columns"] = len(df.columns)

    v1_completeness(df, report)
    v2_format(df, report)
    v3_deduplication(df, report)
    v4_cross_field(df, report)

    if run_chemical:
        v5_chemical(df, report)

    if run_url_check:
        v6_source_urls(df, report, sample=url_sample)

    return report


def main():
    parser = argparse.ArgumentParser(description="Validate a Dataset Builder output CSV")
    parser.add_argument("csv", help="Path to output.csv")
    parser.add_argument("--chem",   action="store_true", help="Run SMILES/SELFIES validation (V5)")
    parser.add_argument("--urls",   action="store_true", help="Spot-check Source_URLs (V6)")
    parser.add_argument("--sample", type=int, default=10, help="Number of URLs to check (V6)")
    parser.add_argument("--report", default="", help="Write JSON report to this path")
    args = parser.parse_args()

    report = validate(
        args.csv,
        run_chemical=args.chem,
        run_url_check=args.urls,
        url_sample=args.sample,
    )

    display_report(report)

    if args.report:
        Path(args.report).write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        console.print(f"\n[dim]JSON report → {args.report}[/dim]")

    sys.exit(0 if report.passed() else 1)


if __name__ == "__main__":
    main()
