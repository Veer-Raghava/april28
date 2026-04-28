"""
agents/schema_discovery.py — Agent 1b: Column Schema Generator.

State Bus Contract
------------------
  INPUT  : data/state/user_requests.jsonl
           Each line: {"topic": str, "columns": list[str], "status": "pending", ...}

  OUTPUT : data/state/schema.json   (single file, overwritten each run)
           {
             "topic": str,
             "columns": [...],
             "field_definitions": {"col_name": "definition..."},
             "field_types": {"col_name": "str|int|float|list|bool"},
             "null_strategy": {"col_name": "crossref|llm_retry|carry_forward|accept"},
             "identity_columns": [...],
             "content_columns": [...],
             "requires_vision": ["SMILES", ...],   # columns needing image/OCR
             "generated_at": float
           }

  LOGS   : data/state/agent_logs.jsonl

Runs concurrently with QueryArchitectAgent. Both read user_requests.jsonl.
ExtractionAgent reads schema.json before processing any document.
"""

from __future__ import annotations

import time
from typing import Any

from tools.console_setup import console
from agents.base_agent import BaseAgent


# Columns whose values are best read from chemical structure images / OCR
_VISION_KEYWORDS = {
    "smiles", "inchi", "structure", "formula", "mol", "chemical",
    "cas", "compound_structure", "molecular",
}

# Columns that are typically numeric
_NUMERIC_KEYWORDS = {
    "dose", "concentration", "ic50", "ec50", "ki", "kd", "pvalue",
    "p_value", "score", "rating", "efficiency", "yield", "amount",
    "quantity", "weight", "size", "count", "number", "age", "year",
}

# Columns well-served by CrossRef API backfill
_CROSSREF_KEYWORDS = {
    "doi", "title", "authors", "author", "journal", "publication",
    "publisher", "year", "date", "abstract",
}


class SchemaDiscoveryAgent(BaseAgent):
    """
    Agent 1b — Generates a strict, typed schema for the requested columns.

    Reads ``user_requests.jsonl``, calls the LLM to produce detailed field
    definitions, expected data types, and null-handling strategies, then
    writes the result to ``schema.json`` (overwritten atomically).

    The ExtractionAgent reads this file before processing any document,
    giving it richer guidance than a simple list of column names.
    """

    def __init__(self, config: dict):
        super().__init__(agent_id="schema_discovery", config=config)

    # ------------------------------------------------------------------ #
    #  Async Run Loop                                                      #
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        """
        1. Reads ``user_requests.jsonl`` for ``status == "pending"``.
        2. For the first pending request, generates a strict schema via LLM.
        3. Writes the schema to ``schema.json`` (overwrites).
        4. Skips if schema.json already exists and is newer than the request.
        """
        console.print("\n[bold cyan]📐 Agent 1b: Schema Discovery[/bold cyan] — starting")

        requests = self.read_input_queue("user_requests.jsonl")
        pending  = [r for r in requests if r.get("status") == "pending"]

        if not pending:
            console.print("  [dim yellow]No pending requests in user_requests.jsonl[/dim yellow]")
            self.log_status("run", "skip", {"reason": "no pending requests"})
            return

        # Process the first pending request (one schema per pipeline run)
        req     = pending[0]
        topic   = req.get("topic", "").strip()
        columns = req.get("columns", [])

        if not topic:
            console.print("  [red]✗ Request missing 'topic' field — cannot generate schema.[/red]")
            self.log_status("run", "error", {"reason": "missing topic"})
            return

        console.print(f"  [bold]Generating schema for:[/bold] '{topic}'")
        console.print(f"  [dim]{len(columns)} column(s) requested[/dim]")
        self.log_status("started", "success", {"topic": topic, "columns": columns})

        # ── LLM: Field definitions ────────────────────────────────────────
        field_definitions = self._generate_field_definitions(columns, topic)

        # ── LLM: Expected data types ──────────────────────────────────────
        field_types = self._generate_field_types(columns, topic)

        # ── LLM: Null-handling strategy ───────────────────────────────────
        null_strategy = self._generate_null_strategy(columns, topic)

        # ── LLM: Identity vs content classification ───────────────────────
        identity_cols, content_cols = self._classify_columns(columns, topic)

        # ── Rule-based: Vision columns (no LLM call needed) ──────────────
        requires_vision = [
            c for c in columns
            if any(kw in c.lower() for kw in _VISION_KEYWORDS)
        ]

        # ── Assemble and write schema ─────────────────────────────────────
        schema: dict = {
            "topic":             topic,
            "columns":           columns,
            "field_definitions": field_definitions,
            "field_types":       field_types,
            "null_strategy":     null_strategy,
            "identity_columns":  identity_cols,
            "content_columns":   content_cols,
            "requires_vision":   requires_vision,
            "generated_at":      time.time(),
        }

        # write_output with .json extension → atomic overwrite
        self.write_output("schema.json", schema)

        self.log_status(
            "completed", "success",
            {
                "topic":           topic,
                "columns":         len(columns),
                "vision_cols":     requires_vision,
                "identity_cols":   identity_cols,
            },
        )
        console.print(
            f"  [green]✓ Schema written to schema.json "
            f"({len(columns)} fields, {len(requires_vision)} vision field(s))[/green]"
        )

    # ------------------------------------------------------------------ #
    #  Private LLM Methods                                                 #
    # ------------------------------------------------------------------ #

    def _generate_field_definitions(self, columns: list[str], topic: str) -> dict[str, str]:
        """
        Generate a precise one-sentence definition for each column,
        including extraction hints and example values in parentheses.
        """
        system = (
            "You are a scientific data schema expert. "
            "Return a JSON object mapping each column name (lowercase) to a precise, "
            "one-sentence definition of what data to extract. "
            "Include expected format and 1-2 examples in parentheses. "
            "Be specific to the research topic. "
            "Return ONLY a JSON object, no prose."
        )
        col_list = ", ".join(f'"{c}"' for c in columns[:15])
        result = self.llm.complete_json(
            f'Topic: "{topic}"\nColumns: [{col_list}]\n\nReturn field definitions:',
            system=system, max_tokens=1500,
        )
        if isinstance(result, dict):
            return {str(k).lower(): str(v) for k, v in result.items()}
        # Fallback: stub definitions
        return {c.lower(): f"The {c} value extracted from the text about {topic}." for c in columns}

    def _generate_field_types(self, columns: list[str], topic: str) -> dict[str, str]:
        """
        Classify each column's expected Python type.
        Valid types: "str", "int", "float", "list", "bool".
        """
        system = (
            "You are a data schema expert. "
            'Return a JSON object mapping each column name to its expected data type. '
            'Valid types: "str", "int", "float", "list", "bool". '
            "Consider the research domain carefully. "
            "Return ONLY a JSON object, no prose."
        )
        col_list = ", ".join(f'"{c}"' for c in columns[:15])
        result = self.llm.complete_json(
            f'Topic: "{topic}"\nColumns: [{col_list}]\n\nReturn field types:',
            system=system, max_tokens=512,
        )
        if isinstance(result, dict):
            valid_types = {"str", "int", "float", "list", "bool"}
            typed = {}
            for c in columns:
                raw = str(result.get(c, result.get(c.lower(), "str"))).lower()
                typed[c] = raw if raw in valid_types else "str"
            return typed
        # Rule-based fallback
        typed = {}
        for c in columns:
            cl = c.lower()
            if any(kw in cl for kw in _NUMERIC_KEYWORDS):
                typed[c] = "float"
            else:
                typed[c] = "str"
        return typed

    def _generate_null_strategy(self, columns: list[str], topic: str) -> dict[str, str]:
        """
        Assign a null-handling strategy to each column.

        Strategies
        ----------
        "crossref"       — Backfill via CrossRef API (DOI, title, authors, year, journal).
        "carry_forward"  — Copy best value from same source across rows.
        "llm_retry"      — Run a focused re-extraction LLM call (expensive).
        "vision"         — Requires image/OCR tool (placeholder for future).
        "accept"         — Accept N/A; this field is optional.
        """
        system = (
            "You are a data quality expert for scientific datasets. "
            "Assign a null-handling strategy to each column. "
            "Valid strategies: "
            '"crossref" (for DOI/title/authors/year/journal fields recoverable from CrossRef API), '
            '"carry_forward" (for stable metadata constant across rows from same source), '
            '"llm_retry" (for important columns needing focused LLM re-extraction), '
            '"accept" (for optional/rare fields where N/A is acceptable). '
            "Return ONLY a JSON object mapping column names to strategy strings."
        )
        col_list = ", ".join(f'"{c}"' for c in columns[:15])
        result = self.llm.complete_json(
            f'Topic: "{topic}"\nColumns: [{col_list}]\n\nAssign null strategies:',
            system=system, max_tokens=512,
        )

        valid_strategies = {"crossref", "carry_forward", "llm_retry", "vision", "accept"}

        if isinstance(result, dict):
            strategies = {}
            for c in columns:
                raw = str(result.get(c, result.get(c.lower(), "accept"))).lower()
                strategies[c] = raw if raw in valid_strategies else "accept"

            # Override with vision for known chemical columns
            for c in columns:
                if any(kw in c.lower() for kw in _VISION_KEYWORDS):
                    strategies[c] = "vision"
            return strategies

        # Rule-based fallback
        strategies = {}
        for c in columns:
            cl = c.lower()
            if any(kw in cl for kw in _CROSSREF_KEYWORDS):
                strategies[c] = "crossref"
            elif any(kw in cl for kw in {"title", "authors", "year", "journal", "institution"}):
                strategies[c] = "carry_forward"
            elif any(kw in cl for kw in _VISION_KEYWORDS):
                strategies[c] = "vision"
            else:
                strategies[c] = "llm_retry"
        return strategies

    def _classify_columns(self, columns: list[str], topic: str) -> tuple[list[str], list[str]]:
        """
        Classify columns as identity (uniquely ID an entity) or
        content (describes details about the entity).

        Returns
        -------
        tuple[list[str], list[str]]
            (identity_columns, content_columns)
        """
        system = (
            "Classify each column as 'identity' (uniquely identifies the entity, "
            "e.g. drug name, compound ID, gene symbol) or 'content' (describes "
            "attributes of the entity, e.g. year, dosage, outcome, method). "
            'Return a JSON object with two keys: "identity" and "content", '
            "each containing a list of column names. "
            "Return ONLY the JSON object."
        )
        col_list = ", ".join(f'"{c}"' for c in columns)
        result = self.llm.complete_json(
            f'Topic: "{topic}"\nColumns: [{col_list}]\n\nClassify columns:',
            system=system, max_tokens=512,
        )
        if isinstance(result, dict):
            identity = [str(c) for c in result.get("identity", [])]
            content  = [str(c) for c in result.get("content", [])]
            # Ensure every column appears in exactly one list
            classified = set(identity) | set(content)
            for c in columns:
                if c not in classified:
                    content.append(c)
            return identity, content

        # Rule-based fallback
        id_kw = {"name", "id", "title", "drug", "compound", "gene", "target", "identifier"}
        identity = [c for c in columns if any(kw in c.lower() for kw in id_kw)]
        content  = [c for c in columns if c not in identity]
        return identity, content
