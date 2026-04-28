"""
agents/orchestrator.py — Async Lifecycle Manager.

Architecture
------------
This file no longer orchestrates data flow.  It is purely a *process
lifecycle manager*:

  1. Writes the initial ``user_requests.jsonl`` entry to the state bus.
  2. Instantiates every agent with the shared ``config`` dict.
  3. Runs agents concurrently with ``asyncio.gather()``.
  4. Inspects ``validated_rows.jsonl`` after the gather completes to decide
     whether an adaptive loop is needed (high null rate / insufficient rows).
  5. If so, uses QueryArchitect + RetrievalCoordinator to inject more sources
     and triggers a second gather over Ingestion → Extraction → NullHunter
     → Validator.

Data flow is entirely mediated by JSONL files in ``state_dir``.  No agent
object passes data to another agent object.

Concurrent execution map (asyncio.gather)
-----------------------------------------
  Phase A (concurrent):
    QueryArchitectAgent   — writes queries.jsonl
    SchemaDiscoveryAgent  — writes schema.json

  Phase B (after A):
    RetrievalCoordinatorAgent — writes sources.jsonl

  Phase C (after B):
    IngestionAgent        — writes documents.jsonl

  Phase D (after C, concurrent):
    ExtractionAgent       — writes extracted_rows.jsonl
    NullHunterAgent       — enriches extracted_rows.jsonl (runs after extraction)

  Phase E (after D):
    ValidatorAssemblerAgent — writes final CSV + validated_rows.jsonl
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from tools.console_setup import console
from rich.panel import Panel

import config as cfg
from tools.browser import close_browser
from tools.export import display_final_summary

from agents.query_architect       import QueryArchitectAgent
from agents.schema_discovery      import SchemaDiscoveryAgent
from agents.retrieval_coordinator import RetrievalCoordinatorAgent
from agents.ingestion_agent       import IngestionAgent
from agents.extraction_agent      import ExtractionAgent
from agents.null_hunter           import NullHunterAgent
from agents.validator_assembler   import ValidatorAssemblerAgent

# ── New tools ─────────────────────────────────────────────────────────────────
from tools.auth             import require_auth, AuthError
from tools.validate_dataset import validate as validate_output, display_report


class Orchestrator:
    """
    Async lifecycle manager for the 7-agent pipeline.

    Does NOT touch SharedState or pass data between agents in memory.
    All coordination is done by checking state-bus files and re-invoking
    agent ``run()`` coroutines.
    """

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            Must include:
              ``state_dir``   — shared JSONL bus directory (str | Path).
              ``output_path`` — final CSV path (str).
              ``max_sources`` — int.
              ``min_rows``    — int  target row count.
              ``max_adaptive_loops`` — int.
            Any additional keys are forwarded to agents.
        """
        self.config = config
        self.state_dir   = Path(config["state_dir"])
        self.output_path = config.get("output_path", "data/outputs/output.csv")
        self.min_rows    = int(config.get("min_rows", 50))
        self.max_loops   = int(config.get("max_adaptive_loops", cfg.MAX_ADAPTIVE_LOOPS))

        # Instantiate all agents — each gets the full config dict
        self.query_architect        = QueryArchitectAgent(config)
        self.schema_discovery       = SchemaDiscoveryAgent(config)
        self.retrieval_coordinator  = RetrievalCoordinatorAgent(config)
        self.ingestion_agent        = IngestionAgent(config)
        self.extraction_agent       = ExtractionAgent(config)
        self.null_hunter            = NullHunterAgent(config)
        self.validator_assembler    = ValidatorAssemblerAgent(config)

    # ------------------------------------------------------------------ #
    #  Public Entry Point                                                  #
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        """
        Async pipeline entry point.

        Phases are separated by ``await`` so each phase completes before
        the next begins.  Within each phase, independent agents run via
        ``asyncio.gather()``.
        """
        start_time = time.time()
        console.print(Panel(
            "[bold green]Multi-Agent Dataset Builder — Agentic Mode[/bold green]\n"
            f"  state_dir   : {self.state_dir}\n"
            f"  output_path : {self.output_path}\n"
            f"  min_rows    : {self.min_rows}   max_loops: {self.max_loops}",
            expand=False,
        ))

        # ── Phase A: Planning (concurrent) ───────────────────────────────
        console.print("\n[bold]Phase A — Planning (QueryArchitect + SchemaDiscovery)[/bold]")
        await asyncio.gather(
            self.query_architect.run(),
            self.schema_discovery.run(),
        )

        # ── Adaptive loop ─────────────────────────────────────────────────
        metrics: dict = {}
        for loop in range(self.max_loops):
            console.print(f"\n[bold]──── Adaptive Loop {loop + 1}/{self.max_loops} ────[/bold]")

            # ── Phase B: Retrieval ────────────────────────────────────────
            console.print("\n[bold]Phase B — Retrieval Coordinator[/bold]")
            await self.retrieval_coordinator.run()

            if not self._has_pending("sources.jsonl", "pending_ingestion"):
                console.print("[red]No sources passed triage — stopping.[/red]")
                break

            # ── Phase C: Ingestion ────────────────────────────────────────
            console.print("\n[bold]Phase C — Ingestion[/bold]")
            await self.ingestion_agent.run()

            if not self._has_pending("documents.jsonl", "pending_extraction"):
                console.print("[red]No documents ingested — stopping.[/red]")
                break

            # ── Phase D: Extraction then NullHunter (sequential within phase) ─
            # ExtractionAgent runs first (produces extracted_rows.jsonl),
            # then NullHunter enriches it.  They share the same file so must
            # run sequentially, not concurrently.
            console.print("\n[bold]Phase D — Extraction → Null Hunter[/bold]")
            await self.extraction_agent.run()
            await self.null_hunter.run()

            # ── Phase E: Validation + Assembly ───────────────────────────
            console.print("\n[bold]Phase E — Validator + Assembler[/bold]")
            metrics = await self.validator_assembler.run() or {}

            # ── Source URL integrity check ────────────────────────────────
            self._verify_source_urls()

            # ── Stopping conditions ───────────────────────────────────────
            accepted = metrics.get("accepted", 0)
            if accepted >= self.min_rows:
                console.print(f"\n[green]✓ Target met: {accepted} rows ≥ {self.min_rows}[/green]")
                break

            if not metrics.get("needs_more_sources", False):
                console.print(
                    f"\n[green]✓ Quality threshold met "
                    f"(null rate: {metrics.get('null_rate', 0):.0%})[/green]"
                )
                break

            if loop < self.max_loops - 1:
                # ── Adaptive re-search ────────────────────────────────────
                console.print("\n[yellow]⚠ Need more data — generating fallback queries…[/yellow]")
                gap_cols    = self._find_gap_columns(metrics.get("null_rates", {}))
                queries_rec = self._read_first_query_record()

                if queries_rec:
                    new_queries = self.query_architect.generate_fallback_queries(
                        topic           = queries_rec.get("topic", ""),
                        existing_queries= queries_rec.get("search_queries", []),
                        gap_columns     = gap_cols,
                    )
                    added = self.retrieval_coordinator.add_more_sources(
                        new_queries = new_queries,
                        topic       = queries_rec.get("topic", ""),
                        record      = queries_rec,
                    )
                    if added == 0:
                        console.print("[yellow]No new sources found — stopping.[/yellow]")
                        break
                else:
                    console.print("[yellow]Could not read query record for fallback — stopping.[/yellow]")
                    break

        # ── Cleanup ───────────────────────────────────────────────────────
        close_browser()

        elapsed = time.time() - start_time
        accepted = metrics.get("accepted", 0)
        rejected = metrics.get("rejected", 0)

        display_final_summary(
            accepted    = accepted,
            rejected    = rejected,
            blocked     = self._count_blocked(),
            sources     = self._count_ingested(),
            iterations  = min(loop + 1, self.max_loops),
            null_rate   = metrics.get("null_rate", 0.0),
            elapsed     = elapsed,
            output_path = self.output_path,
        )

        # ── Post-run validation ───────────────────────────────────────────
        if Path(self.output_path).exists():
            console.print("\n[bold]Post-run Validation[/bold]")
            try:
                report = validate_output(
                    self.output_path,
                    run_chemical=cfg.ENABLE_VISION,   # full chem check only when vision active
                    run_url_check=False,              # skip live HEAD checks in automated runs
                )
                display_report(report)
            except Exception as e:
                console.print(f"  [dim yellow]Validation skipped: {e}[/dim yellow]")

    # ------------------------------------------------------------------ #
    #  State Bus Inspection Helpers                                        #
    # ------------------------------------------------------------------ #

    def _has_pending(self, filename: str, status: str) -> bool:
        """Return True if ``filename`` contains at least one record with ``status``."""
        records = self._read_jsonl(filename)
        return any(r.get("status") == status for r in records)

    def _read_jsonl(self, filename: str) -> list[dict]:
        """Read all records from a JSONL file in state_dir. Returns [] on error."""
        filepath = self.state_dir / filename
        if not filepath.exists():
            return []
        records = []
        with filepath.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                except json.JSONDecodeError:
                    pass
        return records

    def _read_first_query_record(self) -> dict | None:
        """Return the first record from queries.jsonl (architect output)."""
        records = self._read_jsonl("queries.jsonl")
        return records[0] if records else None

    def _count_blocked(self) -> int:
        return len(self._read_jsonl("triage_blocked.jsonl"))

    def _count_ingested(self) -> int:
        return len(self._read_jsonl("documents.jsonl"))

    # ------------------------------------------------------------------ #
    #  Source URL Integrity Check                                          #
    # ------------------------------------------------------------------ #

    def _verify_source_urls(self) -> None:
        """
        Strip validated rows whose Source_URL was not in the session's
        approved sources.jsonl list.  Protects against LLM-hallucinated URLs.

        Moves stripped rows from validated_rows.jsonl to rejected_rows.jsonl.
        """
        approved_urls: set[str] = {
            r.get("url", "") for r in self._read_jsonl("sources.jsonl")
        }
        if not approved_urls:
            return

        validated  = self._read_jsonl("validated_rows.jsonl")
        clean      = []
        stripped   = []

        for row in validated:
            url = row.get("Source_URL", "")
            if not url or url in approved_urls:
                clean.append(row)
            else:
                row["_issues"] = row.get("_issues", "") + ";integrity:unverified_url"
                stripped.append(row)

        if stripped:
            # Rewrite validated_rows.jsonl without the stripped rows
            vpath = self.state_dir / "validated_rows.jsonl"
            vpath.write_text("", encoding="utf-8")
            for row in clean:
                line = json.dumps(row, ensure_ascii=False) + "\n"
                with vpath.open("a", encoding="utf-8") as fh:
                    fh.write(line)

            # Append stripped rows to rejected_rows.jsonl
            rpath = self.state_dir / "rejected_rows.jsonl"
            with rpath.open("a", encoding="utf-8") as fh:
                for row in stripped:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")

            console.print(
                f"  [yellow]⚠ Source integrity: stripped {len(stripped)} row(s) "
                f"with unverified URLs[/yellow]"
            )

    # ------------------------------------------------------------------ #
    #  Gap Column Detection                                                #
    # ------------------------------------------------------------------ #

    def _find_gap_columns(self, null_rates: dict[str, float]) -> list[str]:
        """Return the top-3 columns with > 30% null rate for fallback queries."""
        sorted_cols = sorted(null_rates.items(), key=lambda x: -x[1])
        return [col for col, rate in sorted_cols[:3] if rate > 0.3]


# ── Convenience function (used by main.py / chatbot.py) ──────────────────────

def build_config(
    topic: str,
    columns: list[str],
    dataset_name: str = "",
    min_rows: int = 50,
    max_sources: int | None = None,
    max_adaptive_loops: int | None = None,
    output_path: str = "",
) -> dict:
    """
    Build the shared ``config`` dict that every agent and the Orchestrator
    accept.  Initialises the isolated workspace directory via cfg.apply_topic_paths.

    Parameters
    ----------
    topic : str
        Research topic / pipeline description.
    columns : list[str]
        Desired output columns.
    dataset_name : str
        Human-readable dataset label (defaults to topic[:40]).
    min_rows : int
        Minimum accepted rows before stopping.
    max_sources : int | None
        How many sources to approve per round (defaults to cfg.DEFAULT_LIMIT).
    max_adaptive_loops : int | None
        Maximum feedback iterations (defaults to cfg.MAX_ADAPTIVE_LOOPS).
    output_path : str
        Override the CSV output path (empty = auto from topic workspace).

    Returns
    -------
    dict
        Config ready to pass to ``Orchestrator(config)``.
    """
    paths = cfg.apply_topic_paths(topic)

    final_output_path = output_path or paths["output_file"]

    return {
        # Workspace
        "state_dir":           paths["sources_dir"],   # shared bus directory
        "output_path":         final_output_path,

        # Pipeline parameters
        "topic":               topic,
        "dataset_name":        dataset_name or topic[:40],
        "columns":             columns,
        "min_rows":            min_rows,
        "max_sources":         max_sources or cfg.DEFAULT_LIMIT,
        "max_adaptive_loops":  max_adaptive_loops or cfg.MAX_ADAPTIVE_LOOPS,

        # Workspace paths (for informational use)
        "base_dir":   str(paths["base_dir"]),
        "output_dir": str(paths["output_dir"]),
        "chunks_dir": str(paths["chunks_dir"]),
    }


# ── Compat shim ───────────────────────────────────────────────────────────────

class PipelineResult:
    """
    Lightweight result object returned by ``run_pipeline_from_config``.

    Provides the same attribute surface that ``main.py`` and ``chatbot.py``
    previously expected from ``SharedState``, but is populated from the
    state-bus JSONL files after the pipeline runs.
    """

    def __init__(
        self,
        output_path: str = "",
        accepted_rows: list = None,
        rejected_rows: list = None,
        blocked_sources: list = None,
        documents: list = None,
        null_rate: float = 0.0,
        triage_passed: list = None,
    ):
        self.output_path    = output_path
        self.accepted_rows  = accepted_rows  or []
        self.rejected_rows  = rejected_rows  or []
        self.blocked_sources= blocked_sources or []
        self.documents      = documents      or []
        self.null_rate      = null_rate
        self.triage_passed  = triage_passed  or []


def run_pipeline_from_config(
    config: dict,
    client=None,           # kept for API compat — no longer used
    local_pdfs: list[str] | None = None,
) -> PipelineResult:
    """
    Synchronous compat wrapper used by ``main.py`` and ``chatbot.py``.

    Converts a legacy ``config`` dict into ``run_pipeline_async`` parameters,
    drives the event loop, then reads the resulting state-bus files to build
    and return a ``PipelineResult``.

    Parameters
    ----------
    config : dict
        Accepts both the old ``dataset_name`` / ``description`` format and
        the new ``topic`` / ``columns`` format.
    client : LLMClient | None
        Ignored — kept only for backward compatibility.
    local_pdfs : list[str] | None
        Optional local PDF paths injected into sources.jsonl.
    """
    import asyncio
    import json as _json
    from pathlib import Path as _Path

    # ── Map legacy config keys → new format ──────────────────────────────────
    topic = config.get("topic") or config.get("description") or config.get("dataset_name", "")

    raw_cols = config.get("columns", [])
    columns: list[str] = []
    for c in raw_cols:
        if isinstance(c, dict):
            columns.append(c.get("name", ""))
        else:
            columns.append(str(c))
    columns = [c for c in columns if c]

    # ── Run async pipeline ────────────────────────────────────────────────────
    asyncio.run(run_pipeline_async(
        topic               = topic,
        columns             = columns,
        dataset_name        = config.get("dataset_name", ""),
        min_rows            = int(config.get("min_rows", 50)),
        max_sources         = config.get("max_sources"),
        max_adaptive_loops  = config.get("max_adaptive_loops"),
        output_path         = config.get("output_path", ""),
        local_pdfs          = local_pdfs,
    ))

    # ── Read results from state bus ───────────────────────────────────────────
    built_config = build_config(
        topic               = topic,
        columns             = columns,
        dataset_name        = config.get("dataset_name", ""),
        min_rows            = int(config.get("min_rows", 50)),
        max_sources         = config.get("max_sources"),
        max_adaptive_loops  = config.get("max_adaptive_loops"),
        output_path         = config.get("output_path", ""),
    )
    state_dir   = _Path(built_config["state_dir"])
    output_path = built_config["output_path"]

    def _read_jsonl(fname: str) -> list[dict]:
        p = state_dir / fname
        if not p.exists():
            return []
        rows = []
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
                except _json.JSONDecodeError:
                    pass
        return rows

    validated  = _read_jsonl("validated_rows.jsonl")
    rejected   = _read_jsonl("rejected_rows.jsonl")
    blocked    = _read_jsonl("triage_blocked.jsonl")
    documents  = _read_jsonl("documents.jsonl")
    sources    = _read_jsonl("sources.jsonl")

    # Null-rate estimate from validated rows
    if validated and columns:
        from tools.llm_client import is_empty
        data_cols = [c for c in columns if c not in ("Source_URL", "Dataset_Links", "row_confidence")]
        total     = len(validated) * max(len(data_cols), 1)
        nulls     = sum(1 for r in validated for c in data_cols if is_empty(r.get(c, "N/A")))
        null_rate = nulls / total if total else 0.0
    else:
        null_rate = 0.0

    return PipelineResult(
        output_path    = output_path,
        accepted_rows  = validated,
        rejected_rows  = rejected,
        blocked_sources= blocked,
        documents      = documents,
        null_rate      = null_rate,
        triage_passed  = [s.get("url", "") for s in sources],
    )


async def run_pipeline_async(
    topic: str,
    columns: list[str],
    dataset_name: str = "",
    min_rows: int = 50,
    max_sources: int | None = None,
    max_adaptive_loops: int | None = None,
    output_path: str = "",
    local_pdfs: list[str] | None = None,
) -> None:
    """
    Convenience coroutine to build config, seed the state bus, and run the
    full pipeline.  Call with ``asyncio.run(run_pipeline_async(...))``.

    Parameters
    ----------
    local_pdfs : list[str] | None
        Optional list of local PDF paths to inject directly into sources.jsonl,
        bypassing the retrieval step.
    """
    config = build_config(
        topic               = topic,
        columns             = columns,
        dataset_name        = dataset_name,
        min_rows            = min_rows,
        max_sources         = max_sources,
        max_adaptive_loops  = max_adaptive_loops,
        output_path         = output_path,
    )

    # ── Auth gate (optional — only when ENABLE_AUTH=true in .env) ────────
    if cfg.ENABLE_AUTH:
        try:
            require_auth()   # interactive by default; raises AuthError on failure
        except AuthError as exc:
            console.print(f"[bold red]✗ Authentication failed: {exc}[/bold red]")
            raise SystemExit(1) from exc

    state_dir = Path(config["state_dir"])
    state_dir.mkdir(parents=True, exist_ok=True)

    # ── Seed user_requests.jsonl ──────────────────────────────────────────
    request_record = {
        "topic":   topic,
        "columns": columns,
        "status":  "pending",
        "ts":      time.time(),
    }
    req_path = state_dir / "user_requests.jsonl"
    with req_path.open("w", encoding="utf-8") as fh:   # overwrite each run
        fh.write(json.dumps(request_record, ensure_ascii=False) + "\n")

    # ── Inject local PDFs directly into sources.jsonl ────────────────────
    if local_pdfs:
        src_path = state_dir / "sources.jsonl"
        with src_path.open("a", encoding="utf-8") as fh:
            for pdf_path in local_pdfs:
                entry = {
                    "url":    pdf_path,
                    "result": "pdf",
                    "topic":  topic,
                    "status": "pending_ingestion",
                    "ts":     time.time(),
                    "columns": columns,
                }
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        console.print(f"  [dim]📎 Injected {len(local_pdfs)} local PDF(s) into sources.jsonl[/dim]")

    orchestrator = Orchestrator(config)
    await orchestrator.run()
