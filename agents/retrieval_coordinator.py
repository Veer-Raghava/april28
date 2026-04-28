"""
agents/retrieval_coordinator.py — Agent 2: Search + Triage Coordinator.

Key change: max_sources now counts PDF sources only.
HTML sources are collected additionally — they never consume the user's PDF budget.
"""

from __future__ import annotations

import hashlib
import re
import time
from urllib.parse import urlparse, urlencode, parse_qs

from tools.console_setup import console
from agents.base_agent import BaseAgent
from tools.search import search_duckduckgo, configure_domains
from tools.browser import (
    _build_headers, _exponential_backoff_fetch,
    CURL_AVAILABLE, _detect_block_reason,
)
import config as cfg

try:
    from curl_cffi import requests as cffi_requests
except ImportError:
    import requests as cffi_requests

import requests as std_requests

# Review-paper rejection signals
_REVIEW_SIGNALS = [
    "this review", "in this review", "review article", "systematic review",
    "meta-analysis", "we reviewed", "literature review", "narrative review",
    "overview of", "review of the", "comprehensive review", "recent advances",
    "current status", "state of the art", "we summarize",
]
_PRIMARY_SIGNALS = [
    "we measured", "we synthesized", "we prepared", "n=", "n =",
    "patients enrolled", "cell viability", "ic50", "dar =", "dar=",
    "conjugation", "we report", "we describe", "our results",
    "table 1", "table 2", "figure 1", "in vitro", "in vivo",
]


class RetrievalCoordinatorAgent(BaseAgent):
    """
    Agent 2 — Search candidates and pre-screen before ingestion.

    PDF source limit: max_sources is the PDF budget.
    HTML sources pass triage and get ingested too but don't count toward limit.
    Review papers are filtered out at triage.
    """

    DEFAULT_KEYWORDS = [
        "method", "material", "experiment", "result", "finding",
        "conclusion", "discussion", "abstract", "figure", "table",
        "data", "analysis", "study", "protocol",
    ]
    MIN_CONTENT_LENGTH = 500
    MAX_SOURCES_DEFAULT = 10

    def __init__(self, config: dict):
        super().__init__(agent_id="retrieval_coordinator", config=config)
        self.max_sources: int = int(config.get("max_sources", self.MAX_SOURCES_DEFAULT))
        self._seen_hashes: set[str] = set()

    async def run(self) -> None:
        console.print("\n[bold cyan]🔍 Agent 2: Retrieval Coordinator[/bold cyan] — starting")

        query_records = self.read_input_queue("queries.jsonl")
        pending = [r for r in query_records if r.get("status") == "pending_retrieval"]

        if not pending:
            console.print("  [dim yellow]No pending_retrieval records in queries.jsonl[/dim yellow]")
            self.log_status("run", "skip", {"reason": "no pending_retrieval records"})
            return

        total_passed = 0

        for record in pending:
            topic          = record.get("topic", "")
            search_queries = record.get("search_queries") or [topic]
            domain_config  = record.get("domain_config", {})
            content_kw     = record.get("content_keywords", [])

            console.print(
                f"\n  [bold]Searching:[/bold] '{topic}' "
                f"({len(search_queries)} queries)\n"
                f"  [dim]PDF target: {self.max_sources} sources "
                f"(HTML sources collected additionally)[/dim]"
            )
            self.log_status("started", "success", {"topic": topic})
            configure_domains(domain_config)

            results    = search_duckduckgo(search_queries, limit=self.max_sources * 4)
            candidates = [r["url"] for r in results]
            console.print(f"  [dim]Candidates found: {len(candidates)}[/dim]")

            triage_keywords = self.DEFAULT_KEYWORDS + content_kw
            forwarded_meta = {
                "topic":             topic,
                "columns":           record.get("columns", []),
                "field_definitions": record.get("field_definitions", {}),
                "field_examples":    record.get("field_examples", {}),
                "merge_config":      record.get("merge_config", {}),
                "paywall_domains":   record.get("paywall_domains", []),
                "paywall_signals":   record.get("paywall_signals", []),
                "section_headers":   record.get("section_headers", []),
                "url_patterns":      record.get("url_patterns", {}),
            }

            pdf_passed  = 0
            html_passed = 0

            for url in candidates:
                # PDF budget is the hard limit; HTML has no cap
                if pdf_passed >= self.max_sources:
                    break

                canon = self._canonicalize_url(url)
                if canon in self._seen_hashes:
                    continue
                self._seen_hashes.add(canon)

                is_pdf_url = url.lower().endswith(".pdf") or "/pdf/" in url.lower()
                triage_result = self._triage_url(url, triage_keywords)

                if triage_result in ("pass", "pdf"):
                    # PDF counting logic
                    if triage_result == "pdf" or is_pdf_url:
                        if pdf_passed >= self.max_sources:
                            continue   # PDF budget exhausted
                        pdf_passed += 1
                        icon = "📄"
                    else:
                        html_passed += 1
                        icon = "🌐"

                    self.write_output("sources.jsonl", {
                        "url":         url,
                        "result":      triage_result,
                        "source_hash": canon,
                        "is_pdf":      triage_result == "pdf" or is_pdf_url,
                        "status":      "pending_ingestion",
                        "ts":          time.time(),
                        **forwarded_meta,
                    })
                    total_passed += 1
                    console.print(
                        f"  [green]{icon} Triage {triage_result}: {url[:70]}[/green]"
                    )

                elif triage_result.startswith("blocked:"):
                    reason = triage_result.split(":", 1)[1]
                    self.write_output("triage_blocked.jsonl", {
                        "url": url, "reason": reason, "topic": topic, "ts": time.time(),
                    })
                    console.print(f"  [yellow]🛡 Blocked ({reason}): {url[:60]}[/yellow]")

                elif triage_result == "review_paper":
                    self.write_output("triage_blocked.jsonl", {
                        "url": url, "reason": "review_paper", "topic": topic, "ts": time.time(),
                    })
                    console.print(f"  [yellow]📖 Review paper filtered: {url[:60]}[/yellow]")

                else:
                    console.print(f"  [dim]✗ Triage fail ({triage_result}): {url[:60]}[/dim]")

            self.log_status("completed", "success", {
                "topic": topic, "pdf_passed": pdf_passed,
                "html_passed": html_passed, "candidates": len(candidates),
            })
            console.print(
                f"  [green]✓ Retrieval complete: {pdf_passed} PDF(s) + "
                f"{html_passed} HTML(s) approved[/green]"
            )

        console.print(
            f"\n[bold cyan]🔍 Retrieval Coordinator[/bold cyan] — "
            f"done. {total_passed} total URL(s) written to sources.jsonl."
        )

    def add_more_sources(self, new_queries: list[str], topic: str, record: dict) -> int:
        console.print(f"\n[bold cyan]🔍 Retrieval Coordinator: Additional Search[/bold cyan]")
        self.log_status("feedback_search", "success", {"new_queries": len(new_queries)})

        results    = search_duckduckgo(new_queries, limit=self.max_sources * 2)
        added      = 0
        triage_kw  = self.DEFAULT_KEYWORDS + record.get("content_keywords", [])

        for r in results:
            url   = r["url"]
            canon = self._canonicalize_url(url)
            if canon in self._seen_hashes:
                continue
            self._seen_hashes.add(canon)

            result = self._triage_url(url, triage_kw)
            if result in ("pass", "pdf"):
                self.write_output("sources.jsonl", {
                    "url":         url,
                    "result":      result,
                    "source_hash": canon,
                    "is_pdf":      result == "pdf",
                    "status":      "pending_ingestion",
                    "ts":          time.time(),
                    **{k: record.get(k) for k in [
                        "topic", "columns", "field_definitions", "field_examples",
                        "merge_config", "paywall_domains", "paywall_signals",
                        "section_headers", "url_patterns",
                    ]},
                })
                added += 1
                if added >= self.max_sources:
                    break

        console.print(f"  [green]✓ Added {added} additional source(s)[/green]")
        return added

    def _canonicalize_url(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)
            for p in ["utm_source", "utm_medium", "utm_campaign", "ref", "fbclid"]:
                qs.pop(p, None)
            clean = parsed._replace(
                query=urlencode(qs, doseq=True) if qs else "", fragment="",
            ).geturl().rstrip("/").lower()
            return hashlib.md5(clean.encode()).hexdigest()
        except Exception:
            return hashlib.md5(url.encode()).hexdigest()

    def _is_likely_review(self, snippet: str) -> bool:
        s = snippet.lower()
        primary_hits = sum(1 for sig in _PRIMARY_SIGNALS if sig in s)
        if primary_hits >= 3:
            return False
        review_hits = sum(1 for sig in _REVIEW_SIGNALS if sig in s)
        return review_hits >= 2

    def _triage_url(self, url: str, keywords: list[str]) -> str:
        if url.lower().endswith(".pdf") or "/pdf/" in url.lower():
            return "pdf"
        try:
            headers = _build_headers(url)
            if CURL_AVAILABLE:
                head = cffi_requests.head(
                    url, headers=headers, timeout=10,
                    allow_redirects=True, impersonate=cfg.CURL_IMPERSONATE,
                )
            else:
                head = std_requests.head(url, headers=headers, timeout=8, allow_redirects=True)

            ct = head.headers.get("Content-Type", "").lower()
            if "pdf" in ct:
                return "pdf"
            if "html" not in ct and "text" not in ct:
                return "not_html"
            cl = head.headers.get("Content-Length")
            if cl and int(cl) < 2000:
                return "too_short"

            resp = _exponential_backoff_fetch(url, headers=headers, max_retries=1)
            if not resp:
                return "error"
            if resp.status_code in (403, 429, 503):
                block = _detect_block_reason(
                    resp.text[:3000] if hasattr(resp, "text") else "",
                    resp.status_code,
                )
                return f"blocked:{block or resp.status_code}"

            snippet = ""
            if hasattr(resp, "text"):
                snippet = resp.text[:8000].lower()
            elif hasattr(resp, "content"):
                snippet = resp.content[:8000].decode("utf-8", errors="ignore").lower()

            block = _detect_block_reason(snippet)
            if block:
                return f"blocked:{block}"

            if not any(kw in snippet for kw in keywords):
                return "no_content"

            # Review paper filter
            if self._is_likely_review(snippet):
                return "review_paper"

            return "pass"
        except Exception:
            return "error"
