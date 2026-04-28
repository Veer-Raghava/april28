"""
agents/query_architect.py — Agent 1: Topic Decomposition → Full Config.

Agentic Architecture
--------------------
This agent reads pending user requests from the state bus (user_requests.jsonl),
generates all topic-specific configuration via LLM calls, and writes the
unified output to queries.jsonl for downstream agents to consume.

State Bus Contract
------------------
  INPUT  : data/state/user_requests.jsonl
           Each line: {"topic": str, "columns": list[str], "status": "pending", ...}

  OUTPUT : data/state/queries.jsonl
           Each line: {"topic": str, "sub_topics": [...], "search_queries": [...],
                       "domain_config": {...}, ..., "status": "pending_retrieval"}

  LOGS   : data/state/agent_logs.jsonl  (via BaseAgent.log_status)

Generates all topic-specific configuration in one coherent unit:
  - Sub-topic decomposition
  - Search queries (per sub-topic)
  - Domain config (relevant, blocked, tiers, paywall)
  - Content keywords
  - Field definitions & examples
  - Merge config (identity vs content columns)
  - Section header patterns
  - URL/identifier patterns
"""

from __future__ import annotations

import re
import time
from typing import Any

from tools.console_setup import console
from agents.base_agent import BaseAgent
import config as cfg
from agents.query_architect_prompt import (
    QUERY_ARCHITECT_SYSTEM,
    build_query_prompt,
    build_fallback_prompt,
    DEFAULT_NUM_QUERIES,
)
from tools.image_chem_extractor import classify_vision_columns


class QueryArchitectAgent(BaseAgent):
    """
    Agent 1 — Runs at pipeline start to configure everything downstream.

    Reads pending requests from ``user_requests.jsonl``, generates the full
    topic configuration via a sequence of focused LLM calls, and emits a
    single unified record to ``queries.jsonl``.
    """

    MIN_QUERIES = 3
    MAX_RETRIES = 2

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            Must include ``"state_dir"`` (path to the shared state bus directory).
            Passed directly to BaseAgent which sets up ``self.state_dir`` and
            ``self.llm``.
        """
        super().__init__(agent_id="query_architect", config=config)

    # ------------------------------------------------------------------ #
    #  Async Run Loop (State Bus entry point)                              #
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        """
        Main agent loop.

        1. Reads all records from ``user_requests.jsonl``.
        2. Processes each record whose ``status`` is ``"pending"``.
        3. Generates the full topic configuration and writes it to
           ``queries.jsonl`` with ``status = "pending_retrieval"``.
        4. Skips records that are already processed or malformed.
        """
        console.print("\n[bold cyan]🧠 Agent 1: Query Architect[/bold cyan] — starting")

        pending_requests = self.read_input_queue("user_requests.jsonl")

        if not pending_requests:
            console.print("  [dim yellow]No pending requests found in user_requests.jsonl[/dim yellow]")
            self.log_status(
                stage="run",
                status="skip",
                details={"reason": "user_requests.jsonl is empty or missing"},
            )
            return

        processed = 0
        for item in pending_requests:
            # Only act on explicitly pending records.
            if item.get("status") != "pending":
                continue

            topic   = item.get("topic", "").strip()
            columns = item.get("columns", [])
            # Stash columns so _generate_queries can access them without extra param
            self._current_columns = columns

            if not topic:
                console.print("  [yellow]  ⚠ Skipping record with missing 'topic' field.[/yellow]")
                self.log_status(
                    stage="validate_request",
                    status="warning",
                    details={"reason": "missing 'topic' field", "item": item},
                )
                continue

            console.print(f"\n  [bold]Processing topic:[/bold] '{topic}'")
            self.log_status(
                stage="started",
                status="success",
                details={"topic": topic, "columns": columns},
            )

            # ── Step 1: Sub-topic decomposition ──────────────────────────
            sub_topics = self._decompose_topic(topic)
            console.print(f"  [dim]Sub-topics: {sub_topics}[/dim]")
            self.log_status(
                stage="decompose_topic",
                status="success",
                details={"topic": topic, "sub_topics": sub_topics},
            )

            # ── Step 2: Search queries ────────────────────────────────────
            search_queries = self._generate_queries(topic, sub_topics)
            console.print(f"  [dim]Queries generated: {len(search_queries)}[/dim]")
            self.log_status(
                stage="generate_queries",
                status="success",
                details={"topic": topic, "query_count": len(search_queries)},
            )

            # ── Step 3: Domain config (relevant, blocked, tiers, paywall) ─
            domain_config = self._generate_domain_config(topic)
            self.log_status(
                stage="generate_domain_config",
                status="success",
                details={
                    "topic": topic,
                    "relevant_count": len(domain_config.get("relevant_domains", [])),
                    "blocked_count":  len(domain_config.get("blocked_domains", [])),
                },
            )

            # ── Step 4: Content keywords ──────────────────────────────────
            content_keywords = self._generate_content_keywords(topic)
            self.log_status(
                stage="generate_content_keywords",
                status="success",
                details={"topic": topic, "keyword_count": len(content_keywords)},
            )

            # ── Step 5 & 6: Field definitions, examples, merge config ─────
            # (only when the caller supplied output columns)
            field_definitions: dict[str, str] = {}
            field_examples:    dict[str, Any]  = {}
            merge_config:      dict            = {"identity": ["name", "title"], "content": ["description", "year"]}

            if columns:
                field_definitions = self._generate_field_definitions(columns, topic)
                self.log_status(
                    stage="generate_field_definitions",
                    status="success",
                    details={"topic": topic, "columns": columns},
                )

                field_examples = self._generate_field_examples(columns, topic)
                self.log_status(
                    stage="generate_field_examples",
                    status="success",
                    details={"topic": topic, "example_keys": list(field_examples.keys())},
                )

                merge_config = self._generate_merge_config(columns, topic)
                self.log_status(
                    stage="generate_merge_config",
                    status="success",
                    details={"topic": topic, "merge_config_keys": list(merge_config.keys())},
                )

            # ── Step 7: Section headers + URL patterns ────────────────────
            section_headers = self._generate_section_headers(topic)
            url_patterns    = self._generate_url_patterns(topic)
            self.log_status(
                stage="generate_patterns",
                status="success",
                details={
                    "topic":           topic,
                    "header_patterns": len(section_headers),
                    "url_patterns":    len(url_patterns),
                },
            )

            # ── Vision columns: classify which columns need image extraction ─
            vision_cols = classify_vision_columns(columns)
            if vision_cols:
                console.print(f"  [dim]🔬 Vision columns detected: {vision_cols}[/dim]")

            # ── Bundle and write to state bus ─────────────────────────────
            architect_output: dict = {
                # Provenance — original request fields are preserved wholesale
                "user_request": item,

                # Computed config fields
                "topic":             topic,
                "columns":           columns,
                "sub_topics":        sub_topics,
                "search_queries":    search_queries,
                "domain_config":     domain_config,
                "paywall_domains":   domain_config.get("paywall_domains", []),
                "paywall_signals":   domain_config.get("paywall_signals", []),
                "content_keywords":  content_keywords,
                "field_definitions": field_definitions,
                "field_examples":    field_examples,
                "merge_config":      merge_config,
                "section_headers":   section_headers,
                "url_patterns":      url_patterns,
                "requires_vision":   vision_cols,

                # Routing metadata
                "status":       "pending_retrieval",
                "generated_at": time.time(),
            }

            self.write_output("queries.jsonl", architect_output)

            self.log_status(
                stage="completed",
                status="success",
                details={
                    "topic":        topic,
                    "query_count":  len(search_queries),
                    "keyword_count": len(content_keywords),
                },
            )
            console.print(f"  [green]✓ Query Architect complete for '{topic}'[/green]")
            processed += 1

        console.print(
            f"\n[bold cyan]🧠 Query Architect[/bold cyan] — "
            f"done. Processed {processed} request(s)."
        )

    # ------------------------------------------------------------------ #
    #  Fallback Query Generator (called by feedback / null-hunter loops)   #
    # ------------------------------------------------------------------ #

    def generate_fallback_queries(
        self,
        topic: str,
        existing_queries: list[str],
        gap_columns: list[str],
    ) -> list[str]:
        """
        Generate new queries targeting data gaps.

        Unlike the main ``run()`` loop, this can be called directly by the
        Null Hunter or orchestrator when it detects columns with high null
        rates.

        Parameters
        ----------
        topic : str
            The research topic.
        existing_queries : list[str]
            Queries already tried (to avoid duplication).
        gap_columns : list[str]
            Columns that still have excessive null rates.

        Returns
        -------
        list[str]
            Up to 3 new search queries.
        """
        prompt  = build_fallback_prompt(topic, gap_columns, existing_queries, num_queries=5)
        result  = self.llm.complete_json(prompt, system=QUERY_ARCHITECT_SYSTEM, max_tokens=512)
        if isinstance(result, dict):
            queries = result.get("fallback_queries", [])
            if queries:
                return [str(q) for q in queries[:5]]
        if isinstance(result, list):
            return [str(q) for q in result[:5]]
        # Deterministic fallback — always returns something usable
        return [f'{topic} {col} experimental data' for col in gap_columns[:3]]

    # ------------------------------------------------------------------ #
    #  Private LLM Methods (prompts unchanged from original)              #
    # ------------------------------------------------------------------ #

    def _decompose_topic(self, topic: str) -> list[str]:
        system = (
            "You are a research analyst. Decompose the given topic into 3-5 key sub-topics "
            "or aspects that should each be searched separately for comprehensive coverage. "
            "Return ONLY a JSON array of short sub-topic strings."
        )
        result = self.llm.complete_json(
            f'Topic: "{topic}"\n\nDecompose into sub-topics:',
            system=system, max_tokens=256,
        )
        if isinstance(result, list) and result:
            return [str(s) for s in result[:5]]
        return [topic]

    def _generate_queries(self, topic: str, sub_topics: list[str]) -> list[str]:
        """
        Generate search queries using the tiered primary-data prompt from
        query_architect_prompt.py.  Falls back to sub-topic split queries.
        """
        # Read columns from the pending request stored on self (set during run)
        columns = getattr(self, "_current_columns", [])
        num_q   = getattr(cfg, "NUM_QUERIES", DEFAULT_NUM_QUERIES)
        prompt  = build_query_prompt(topic, columns, num_queries=num_q)

        for attempt in range(self.MAX_RETRIES + 1):
            result = self.llm.complete_json(prompt, system=QUERY_ARCHITECT_SYSTEM, max_tokens=2048)
            if isinstance(result, dict):
                queries = result.get("search_queries", [])
                if isinstance(queries, list) and len(queries) >= self.MIN_QUERIES:
                    console.print(f"  [dim]Using tiered prompt queries: {len(queries)} generated[/dim]")
                    return [str(q) for q in queries[:num_q]]
            elif isinstance(result, list) and len(result) >= self.MIN_QUERIES:
                return [str(q) for q in result[:num_q]]
            if attempt < self.MAX_RETRIES:
                console.print(f"  [dim yellow]Retrying query generation (attempt {attempt+2})…[/dim yellow]")

        # Deterministic fallback — split across sub-topics
        return [
            f'{topic} experimental results data table',
            f'{topic} primary research paper doi filetype:pdf',
            f'{topic} measured values assay site:pubmed.ncbi.nlm.nih.gov',
            f'{topic} dataset synthesised tested',
        ]

    def _generate_domain_config(self, topic: str) -> dict:
        system = (
            "You are a web research expert. Generate domain configuration for a web scraper. "
            "Return a JSON object with these keys:\n"
            '"relevant_domains": list of 15-25 domains with quality content\n'
            '"blocked_domains": list of 10-20 domains to skip\n'
            '"priority_tiers": list of 3-4 lists of domains ordered by quality\n'
            '"paywall_domains": list of 5-15 paywalled domain patterns\n'
            '"paywall_signals": list of 8-12 HTML phrases indicating paywalls\n'
            "Return ONLY a JSON object."
        )
        result = self.llm.complete_json(
            f'Topic: "{topic}"\n\nGenerate domain config:',
            system=system, max_tokens=1024,
        )
        if isinstance(result, dict):
            # Self-validate: ensure blocked_domains are bare hostnames, not full URLs
            if "blocked_domains" in result:
                result["blocked_domains"] = [
                    d.replace("https://", "").replace("http://", "").split("/")[0]
                    for d in result["blocked_domains"]
                ]
            return result
        return {
            "relevant_domains": [],
            "blocked_domains": ["wikipedia.org", "reddit.com", "quora.com", "youtube.com"],
            "priority_tiers": [],
            "paywall_domains": [],
            "paywall_signals": ["subscribe to read", "purchase access", "buy this article"],
        }

    def _generate_content_keywords(self, topic: str) -> list[str]:
        system = (
            "Return 10-15 lowercase keywords that appear in high-quality, data-rich "
            "articles about this topic (not homepages or ads). "
            "Return ONLY a JSON array of lowercase strings."
        )
        result = self.llm.complete_json(
            f'Topic: "{topic}"\n\nGenerate content keywords:',
            system=system, max_tokens=256,
        )
        if isinstance(result, list) and result:
            return [str(k).lower() for k in result]
        return ["result", "data", "analysis", "study", "method", "finding", "table"]

    def _generate_field_definitions(self, columns: list[str], topic: str) -> dict[str, str]:
        system = (
            "Return a JSON object mapping each column name (lowercase) to a precise, "
            "one-sentence definition of what data to extract. Include examples in parentheses. "
            "Return ONLY a JSON object."
        )
        col_list = ", ".join(f'"{c}"' for c in columns[:12])
        result = self.llm.complete_json(
            f'Topic: "{topic}"\nColumns: [{col_list}]\n\nReturn field definitions:',
            system=system, max_tokens=1024,
        )
        if isinstance(result, dict):
            return {str(k).lower(): str(v) for k, v in result.items()}
        return {}

    def _generate_field_examples(self, columns: list[str], topic: str) -> dict[str, Any]:
        system = (
            "Return exactly one realistic example JSON object showing a correctly "
            "extracted row. Use plausible real values. Return ONLY a JSON array with one object."
        )
        col_list = ", ".join(f'"{c}"' for c in columns[:10])
        result = self.llm.complete_json(
            f'Topic: "{topic}"\nColumns: [{col_list}]\n\nReturn one example row:',
            system=system, max_tokens=512,
        )
        if isinstance(result, list) and result and isinstance(result[0], dict):
            return result[0]
        return {}

    def _generate_merge_config(self, columns: list[str], topic: str) -> dict:
        system = (
            "Classify each column as 'identity' (uniquely identifies an entity) or "
            "'content' (describes details). Return a JSON object with two keys: "
            '"identity": list of lowercase keywords, "content": list of lowercase keywords.'
        )
        col_list = ", ".join(f'"{c}"' for c in columns)
        result = self.llm.complete_json(
            f'Topic: "{topic}"\nColumns: [{col_list}]\n\nClassify columns:',
            system=system, max_tokens=512,
        )
        if isinstance(result, dict):
            return result
        return {"identity": ["name", "title"], "content": ["description", "year"]}

    def _generate_section_headers(self, topic: str) -> list[str]:
        system = (
            "Return 4-6 regex fragments (lowercase) that match section headings where "
            "documents on this topic list data sources or supplementary materials. "
            "Return ONLY a JSON array."
        )
        result = self.llm.complete_json(
            f'Topic: "{topic}"\n\nGenerate section header patterns:',
            system=system, max_tokens=256,
        )
        if isinstance(result, list):
            valid = []
            for p in result:
                try:
                    re.compile(str(p))
                    valid.append(str(p).lower())
                except re.error:
                    pass  # silently drop invalid regex patterns
            return valid
        return []

    def _generate_url_patterns(self, topic: str) -> dict[str, str]:
        system = (
            "Return a JSON object mapping short labels to regex patterns for URLs or "
            "identifiers linking to data sources for this topic. 3-6 entries. "
            "Return ONLY a JSON object."
        )
        result = self.llm.complete_json(
            f'Topic: "{topic}"\n\nGenerate URL patterns:',
            system=system, max_tokens=256,
        )
        if isinstance(result, dict):
            valid = {}
            for k, v in result.items():
                try:
                    re.compile(str(v))
                    valid[str(k)] = str(v)
                except re.error:
                    pass  # silently drop patterns with invalid regex
            return valid
        return {}
