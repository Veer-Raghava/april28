"""
agents/ingestion_agent.py — Agent 3: Scrape → Unified Document Model.

Agentic Architecture
--------------------
State Bus Contract:
  INPUT  : data/state/triage_passed.jsonl
           Each line: {"url": str, "result": "pass"|"pdf", "topic": str,
                       "status": "pending_ingestion", ...architect metadata...}

  OUTPUT : data/state/documents.jsonl
           Each line: {"source": str, "full_text": str, "tables": [...],
                       "figures": [...], "metadata": {...}, "status": "pending_extraction",
                       ...architect metadata forwarded...}

  LOGS   : data/state/agent_logs.jsonl  (via BaseAgent.log_status)

Method escalation per URL: curl_cffi → Playwright → OCR.
Blocked/paywalled sources are written to triage_blocked.jsonl (not silently skipped).
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import config as cfg
from agents.base_agent import BaseAgent
from tools.console_setup import console
from tools.browser import (
    init_browser, close_browser, download_pdf, random_ua,
    smart_fetch, FetchResult,
    is_paywalled, html_is_paywalled, configure_paywall,
    STEALTH_JS, simulate_human, _build_headers,
    _exponential_backoff_fetch, CURL_AVAILABLE,
)
from tools.pdf_tools import (
    extract_text as pdf_extract_text,
    extract_text_ocr,
    extract_tables as pdf_extract_tables,
    extract_images,
    chunk_text, clean_text, is_duplicate_pdf,
)
from tools.html_tools import (
    extract_text as html_extract_text,
    extract_tables as html_extract_tables,
    find_dataset_urls,
    find_data_availability_section,
    find_supplement_links,
    find_pdf_links,
    update_patterns,
)


MIN_FULLTEXT_CHARS = 3000


class IngestionAgent(BaseAgent):
    """
    Agent 3 — Scrapes URLs and produces serialisable document records.

    Each consumed ``triage_passed.jsonl`` record is scraped and its content
    written as a flat JSON object to ``documents.jsonl``.  This replaces the
    in-memory ``DocumentObject`` / ``SharedState.documents`` pattern.
    """

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            Must include ``"state_dir"``.
        """
        super().__init__(agent_id="ingestion_agent", config=config)

    # ------------------------------------------------------------------ #
    #  Async Run Loop                                                      #
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        """
        Main agent loop.

        1. Reads ``sources.jsonl`` for records with ``status == "pending_ingestion"``.
        2. Applies per-topic paywall + pattern config from the triage record.
        3. Scrapes the URL using method escalation.
        4. Writes a document record to ``documents.jsonl``.
        """
        console.print("\n[bold cyan]📥 Agent 3: Ingestion[/bold cyan] — starting")

        triage_records = self.read_input_queue("sources.jsonl")
        pending = [r for r in triage_records if r.get("status") == "pending_ingestion"]

        if not pending:
            console.print("  [dim yellow]No pending_ingestion records in sources.jsonl[/dim yellow]")
            self.log_status("run", "skip", {"reason": "no pending records"})
            return

        processed = 0

        for record in pending:
            url    = record.get("url", "")
            topic  = record.get("topic", "")
            result = record.get("result", "pass")   # "pass" | "pdf"

            if not url:
                continue

            console.print(f"\n  [cyan]📄 Ingesting: {url[:80]}[/cyan]")
            self.log_status("started", "success", {"url": url, "topic": topic})

            # Apply topic-specific paywall and pattern config
            self._configure_tools(record)

            # Dispatch based on triage hint
            doc_record = None
            if result == "pdf" or url.lower().endswith(".pdf") or "/pdf/" in url.lower():
                doc_record = self._ingest_pdf_url(url)
            else:
                doc_record = self._ingest_html(url)

            if doc_record:
                # Enrich the document with supplementary metadata
                self._enrich_document(doc_record)

                # Forward architect metadata for downstream agents
                doc_record.update({
                    "topic":             topic,
                    "columns":           record.get("columns", []),
                    "field_definitions": record.get("field_definitions", {}),
                    "field_examples":    record.get("field_examples", {}),
                    "merge_config":      record.get("merge_config", {}),
                    "status":            "pending_extraction",
                    "ts":                time.time(),
                })

                self.write_output("documents.jsonl", doc_record)
                processed += 1
                self.log_status(
                    "completed", "success",
                    {"url": url, "chars": len(doc_record.get("full_text", ""))},
                )
                console.print(
                    f"  [green]✓ Ingested ({doc_record.get('method', '?')}): "
                    f"{len(doc_record.get('full_text', '')):,} chars[/green]"
                )
            else:
                self.log_status("failed", "error", {"url": url, "topic": topic})
                console.print(f"  [red]✗ Failed to ingest: {url[:60]}[/red]")

        console.print(
            f"\n[bold cyan]📥 Ingestion Agent[/bold cyan] — done. "
            f"{processed}/{len(pending)} document(s) written."
        )

    # ------------------------------------------------------------------ #
    #  Local PDF ingestion (used by chatbot / direct file paths)           #
    # ------------------------------------------------------------------ #

    def ingest_local_pdf(self, path: str) -> dict | None:
        """
        Process a local PDF file and return the document record dict.
        Writes result to ``documents.jsonl`` automatically.

        Parameters
        ----------
        path : str
            Absolute or relative path to the PDF file.
        """
        console.print(f"\n  [cyan]📄 Local PDF: {os.path.basename(path)}[/cyan]")
        self.log_status("started", "success", {"source": f"local:{os.path.basename(path)}"})

        if not os.path.isfile(path):
            console.print(f"  [red]✗ File not found: {path}[/red]")
            return None

        text = pdf_extract_text(path)
        if not text or len(text.strip()) < 200:
            console.print("  [yellow]Text too short, trying OCR…[/yellow]")
            text = extract_text_ocr(path)

        if not text or len(text.strip()) < 100:
            console.print(f"  [red]✗ Could not extract text from {path}[/red]")
            return None

        stem   = Path(path).stem
        tables = pdf_extract_tables(path)
        figures = extract_images(path, str(Path(cfg.IMAGE_DIR) / stem))

        doc_record = {
            "source":      path,
            "method":      "local_pdf",
            "full_text":   text,
            "text_blocks": chunk_text(text),
            "tables":      tables,
            "figures":     figures,
            "metadata":    {},
            "html":        None,
        }
        self._enrich_document(doc_record)

        self.log_status(
            "completed", "success",
            {"source": path, "chars": len(text)},
        )
        console.print(f"  [green]✓ Extracted {len(text):,} chars[/green]")
        return doc_record

    # ------------------------------------------------------------------ #
    #  Private — scraping helpers                                          #
    # ------------------------------------------------------------------ #

    def _configure_tools(self, record: dict) -> None:
        """Apply paywall domains/signals and HTML pattern config from the triage record."""
        configure_paywall(
            record.get("paywall_domains", []),
            record.get("paywall_signals", []),
        )
        update_patterns(
            section_headers=record.get("section_headers", []),
            url_patterns=record.get("url_patterns", {}),
        )

    def _ingest_html(self, url: str) -> dict | None:
        """Attempt HTML scrape with method escalation."""

        # URL-level paywall check
        if is_paywalled(url):
            console.print("  [yellow]⊘ Paywalled domain[/yellow]")
            self.write_output("triage_blocked.jsonl", {
                "url": url, "reason": "paywall_domain", "ts": time.time(),
            })
            return None

        time.sleep(random.uniform(*cfg.REQUEST_DELAY))

        fetch_result: FetchResult = smart_fetch(url)

        if fetch_result.blocked:
            console.print(f"  [red]🛡 Blocked ({fetch_result.block_reason})[/red]")
            self.write_output("triage_blocked.jsonl", {
                "url": url, "reason": fetch_result.block_reason,
                "status_code": fetch_result.status_code, "ts": time.time(),
            })
            # Still try to extract if we got partial content
            if not fetch_result.html or len(fetch_result.html) < 1000:
                return None

        html = fetch_result.html
        if not html:
            return self._try_fulltext_fallback(url)

        # HTML-level paywall check
        if html_is_paywalled(html):
            console.print("  [yellow]⊘ Paywall detected in HTML body[/yellow]")
            self.write_output("triage_blocked.jsonl", {
                "url": url, "reason": "paywall_html", "ts": time.time(),
            })
            return None

        # Try to find and download a PDF linked from the HTML
        pdf_links = find_pdf_links(html, url)
        if pdf_links:
            console.print(f"  [cyan]🔗 Found {len(pdf_links)} PDF link(s)[/cyan]")
            for pdf_url in pdf_links[:3]:
                pdf_doc = self._ingest_pdf_url(pdf_url)
                if pdf_doc and len(pdf_doc.get("full_text", "")) >= MIN_FULLTEXT_CHARS:
                    return pdf_doc

        # Fall through to HTML text extraction
        text = html_extract_text(html)
        if not text or len(text) < MIN_FULLTEXT_CHARS:
            return self._try_fulltext_fallback(url)

        html_tables  = html_extract_tables(html)
        table_records = [
            {
                "csv":     csv_str,
                "page":    0,
                "rows":    csv_str.count("\n") + 1,
                "cols":    0,
                "caption": "",
            }
            for csv_str in html_tables
        ]

        return {
            "source":      url,
            "method":      f"html:{fetch_result.method}",
            "full_text":   text,
            "text_blocks": chunk_text(text),
            "tables":      table_records,
            "figures":     [],
            "metadata":    {},
            "html":        html,
        }

    def _ingest_pdf_url(self, url: str) -> dict | None:
        """Download and extract a PDF from a URL."""
        pdf_path = download_pdf(url)

        # Playwright fallback
        if not pdf_path:
            try:
                browser = init_browser()
                context = browser.new_context(
                    user_agent=random_ua(),
                    viewport={"width": random.randint(1200, 1920), "height": random.randint(800, 1080)},
                    locale="en-US",
                )
                page = context.new_page()
                page.add_init_script(STEALTH_JS)
                pdf_path = download_pdf(url, page=page)
                context.close()
            except Exception:
                pass

        if not pdf_path:
            return None

        text = pdf_extract_text(pdf_path)
        if not text:
            return None

        stem = re.sub(r"[^\w\-]", "_", url.split("//")[-1])[:60]
        return {
            "source":      url,
            "method":      "pdf",
            "full_text":   text,
            "text_blocks": chunk_text(text),
            "pdf_path":    pdf_path,
            "tables":      pdf_extract_tables(pdf_path),
            "figures":     extract_images(pdf_path, str(Path(cfg.IMAGE_DIR) / stem)),
            "metadata":    {},
            "html":        None,
        }

    def _try_fulltext_fallback(self, url: str) -> dict | None:
        """Search for open-access alternatives via DOI resolution."""
        doi_m = re.search(r"(10\.\d{4,}/[^\s?#&\"'<>]+)", url)
        doi   = doi_m.group(1).rstrip(".,)") if doi_m else None

        candidates: list[str] = []
        if doi:
            try:
                headers = _build_headers(f"https://doi.org/{doi}")
                if CURL_AVAILABLE:
                    from curl_cffi import requests as cffi_req
                    r = cffi_req.head(
                        f"https://doi.org/{doi}", timeout=8, headers=headers,
                        allow_redirects=True, impersonate=cfg.CURL_IMPERSONATE,
                    )
                else:
                    import requests as std_req
                    r = std_req.head(
                        f"https://doi.org/{doi}", timeout=8, headers=headers,
                        allow_redirects=True,
                    )
                if r.status_code == 200:
                    final_url = str(r.url) if hasattr(r, "url") else url
                    if final_url != url:
                        candidates.append(final_url)
                        candidates.append(final_url.rstrip("/") + "/pdf")
            except Exception:
                pass

        for alt_url in candidates[:3]:
            doc = self._ingest_pdf_url(alt_url)
            if doc and len(doc.get("full_text", "")) >= MIN_FULLTEXT_CHARS:
                self.log_status(
                    "fallback_success", "success",
                    {"original_url": url, "fallback_url": alt_url},
                )
                return doc

        return None

    def _enrich_document(self, doc: dict) -> None:
        """
        Add metadata links and save side-effects (chunks, tables, supplements) to disk.
        Operates in-place on the document dict.
        """
        full_text = doc.get("full_text", "")
        html      = doc.get("html")
        source    = doc.get("source", "")

        doc["dataset_links"]   = find_dataset_urls(full_text)
        doc["data_availability"] = find_data_availability_section(full_text)
        doc["supplement_links"]  = find_supplement_links(html, source) if html else []

        stem = re.sub(
            r"[^\w\-]", "_",
            source.split("//")[-1] if "//" in source else Path(source).stem
        )[:60]

        # Save text chunks to disk for inspection / debugging
        chunks_dir = Path(cfg.CHUNK_TEMP_DIR)
        chunks_dir.mkdir(parents=True, exist_ok=True)
        for idx, chunk in enumerate(doc.get("text_blocks", [])):
            (chunks_dir / f"{stem}_chunk{idx+1:03d}.txt").write_text(chunk, encoding="utf-8")

        # Save extracted tables
        if doc.get("tables"):
            tables_dir = Path(cfg.TABLE_DIR) / stem
            tables_dir.mkdir(parents=True, exist_ok=True)
            for i, tbl in enumerate(doc["tables"]):
                (tables_dir / f"table_{i+1:03d}.csv").write_text(tbl["csv"], encoding="utf-8")

        # Save supplement manifest
        if doc.get("supplement_links"):
            sup_dir = Path(cfg.SUPPLEMENT_DIR)
            sup_dir.mkdir(parents=True, exist_ok=True)
            sup_file = sup_dir / f"{stem}_supplements.json"
            sup_file.write_text(
                json.dumps(doc["supplement_links"], indent=2),
                encoding="utf-8",
            )
