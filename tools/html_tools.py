"""
tools/html_tools.py — HTML text extraction, table extraction, metadata parsing.
"""

import io
import re
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup
from tools.console_setup import console



MIN_CONTENT_LENGTH = 300

# ── Patterns for data availability / dataset URLs ─────────────────────────────

DOI_PATTERN = re.compile(r"10\.\d{4,}/[^\s,;\"'<>]+")
REPO_PATTERNS = {
    "Zenodo":    re.compile(r"zenodo\.org/record[s]?/\d+", re.I),
    "Figshare":  re.compile(r"figshare\.com/\S+", re.I),
    "GitHub":    re.compile(r"github\.com/[\w\-]+/[\w\-]+", re.I),
    "Dryad":     re.compile(r"datadryad\.org/\S+", re.I),
    "Kaggle":    re.compile(r"kaggle\.com/\S+", re.I),
    "OSF":       re.compile(r"osf\.io/\S+", re.I),
}
ACCESSION_PATTERNS = {
    "GEO":        re.compile(r"GSE\d{4,}"),
    "SRA":        re.compile(r"SR[APRX]\d{6,}"),
    "BioProject": re.compile(r"PRJNA?\d+"),
}

# Extra patterns injected by Query Architect
_extra_url_patterns: dict[str, re.Pattern] = {}
_extra_section_headers: list[str] = []


def update_patterns(section_headers: list[str] | None = None,
                    url_patterns: dict[str, str] | None = None) -> None:
    global _extra_section_headers, _extra_url_patterns
    if section_headers:
        _extra_section_headers[:] = [str(h).lower() for h in section_headers]
    if url_patterns:
        compiled = {}
        for label, pattern in url_patterns.items():
            try:
                compiled[str(label)] = re.compile(str(pattern), re.I)
            except re.error:
                pass
        _extra_url_patterns.update(compiled)


def find_dataset_urls(text: str) -> list[dict]:
    """Find DOIs, repository links, and accession numbers in text."""
    found, seen = [], set()
    for m in DOI_PATTERN.finditer(text):
        doi = m.group().rstrip(".")
        if doi not in seen:
            seen.add(doi)
            found.append({"type": "DOI", "value": doi})
    for name, pat in REPO_PATTERNS.items():
        for m in pat.finditer(text):
            u = m.group()
            if u not in seen:
                seen.add(u)
                found.append({"type": name, "value": u})
    for name, pat in ACCESSION_PATTERNS.items():
        for m in pat.finditer(text):
            a = m.group()
            if a not in seen:
                seen.add(a)
                found.append({"type": name, "value": a})
    for name, pat in _extra_url_patterns.items():
        for m in pat.finditer(text):
            v = m.group()
            if v not in seen:
                seen.add(v)
                found.append({"type": name, "value": v})
    return found


def find_data_availability_section(text: str) -> str:
    """Find data availability section in academic text."""
    headers = [
        r"data\s+availab", r"availability\s+of\s+data", r"data\s+access",
        r"code\s+availab", r"supplementary\s+(?:information|materials?|data)",
        r"supporting\s+information",
    ]
    all_headers = headers + [h for h in _extra_section_headers if h not in headers]

    pat = re.compile(
        r"(?:^|\n)\s*(?:" + "|".join(all_headers) + r").*?"
        r"(?=\n\s*(?:references|acknowledgment|author|conflict|funding|appendix)|\Z)",
        re.I | re.DOTALL,
    )
    m = pat.search(text)
    if m:
        return m.group().strip()[:2000]

    tail_start = max(0, int(len(text) * 0.80))
    tail_pat = re.compile(
        r"(?:^|\n)\s*(?:" + "|".join(all_headers) + r").*",
        re.I | re.DOTALL,
    )
    m2 = tail_pat.search(text[tail_start:])
    return m2.group().strip()[:2000] if m2 else ""


def extract_text(html: str) -> str:
    """Extract article body text from HTML, stripping boilerplate."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(["nav", "header", "footer", "script", "style",
                               "aside", "iframe", "noscript", "form",
                               "button", "svg", "img", "label"]):
        tag.decompose()

    noise = re.compile(r"Open in a new tab|Find articles by|Correspondence:|Academic Editor", re.I)
    for n in soup.find_all(string=noise):
        if hasattr(n, 'parent') and n.parent:
            n.parent.decompose()

    # Meta tags
    meta_parts = []
    title_tag = soup.find("title")
    if title_tag:
        meta_parts.append(f"Title: {title_tag.get_text(strip=True)}")
    for meta in soup.find_all("meta"):
        name = meta.get("name", "").lower()
        content = meta.get("content", "")
        if content and name == "citation_abstract":
            meta_parts.append(f"Abstract: {content}")
        elif content and name == "citation_title":
            meta_parts.append(f"Title: {content}")
        elif content and name == "citation_author":
            meta_parts.append(f"Author: {content}")
        elif content and name == "citation_doi":
            meta_parts.append(f"DOI: {content}")
    meta_text = "\n".join(meta_parts)

    # Article content
    article_text = ""
    article_tag = soup.find("article")
    pmc_body = soup.find("div", id="body") or soup.find("div", class_="body")

    if article_tag:
        article_text = article_tag.get_text(separator=" ", strip=True)
    elif pmc_body:
        article_text = pmc_body.get_text(separator=" ", strip=True)

    if article_text:
        article_text = re.sub(r'[ \t]+', ' ', article_text)
        article_text = re.sub(r'([.?!])\s+([A-Z])', r'\1\n\n\2', article_text)

    if not article_text or len(article_text) < MIN_CONTENT_LENGTH:
        for selector in [
            soup.find("main"),
            soup.find("div", class_=re.compile(r"(article|content|paper|fulltext)", re.I)),
            soup.find("div", id=re.compile(r"(article|content|paper|fulltext)", re.I)),
            soup.find("div", role="main"),
        ]:
            if selector:
                candidate = selector.get_text(separator="\n", strip=True)
                if len(candidate) > len(article_text):
                    article_text = candidate
                    break

    if not article_text or len(article_text) < MIN_CONTENT_LENGTH:
        body = soup.find("body")
        article_text = body.get_text(separator="\n", strip=True) if body else soup.get_text(separator="\n", strip=True)

    full_text = (meta_text + "\n\n---\n\n" + article_text) if meta_text else article_text
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    full_text = re.sub(r" {2,}", " ", full_text)
    return full_text


def extract_tables(html: str) -> list[str]:
    """Extract HTML tables as CSV strings."""
    try:
        soup = BeautifulSoup(html, "lxml")
        results = []
        for tbl in soup.find_all("table"):
            try:
                df = pd.read_html(io.StringIO(str(tbl)))[0]
                results.append(df.to_csv(index=False))
            except Exception:
                continue
        return results
    except Exception:
        return []


def find_supplement_links(html: str, base_url: str) -> list[dict]:
    """Find links to supplementary data files."""
    soup = BeautifulSoup(html, "lxml")
    results, seen = [], set()
    DATA_EXTS = {".csv", ".xlsx", ".xls", ".zip", ".tsv", ".json", ".xml"}
    SUP_KEYWORDS = {"supplement", "supporting", "additional", "appendix",
                    "data", "table s", "figure s", "dataset"}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True).lower()
        ext = Path(href).suffix.lower()
        if (any(k in text for k in SUP_KEYWORDS) or ext in DATA_EXTS) and href not in seen:
            seen.add(href)
            if not href.startswith("http"):
                href = urljoin(base_url, href)
            results.append({"url": href, "filename": Path(href).name,
                           "type": ext, "text": a.get_text(strip=True)[:100]})
    return results


def find_pdf_links(html: str, base_url: str) -> list[str]:
    """Find PDF download links from an article HTML page."""
    soup = BeautifulSoup(html, "lxml")
    pdf_urls = []
    seen = set()

    # Meta citation_pdf_url
    meta_pdf = soup.find("meta", attrs={"name": "citation_pdf_url"})
    if meta_pdf and meta_pdf.get("content"):
        pdf_urls.append(meta_pdf["content"])
        seen.add(meta_pdf["content"])

    # URL path heuristics
    base_clean = base_url.rstrip("/")
    for suffix in ["/pdf", "/pdf/", ".pdf"]:
        candidate = base_clean + suffix
        if candidate not in seen and candidate.lower() != base_url.lower():
            seen.add(candidate)
            pdf_urls.append(candidate)

    # URL rewriting
    url_lower = base_url.lower()
    for old, new in [("/abs/", "/pdf/"), ("/view/", "/download/"), ("/article/", "/article/pdf/")]:
        if old in url_lower:
            rewritten = url_lower.replace(old, new, 1)
            if not rewritten.endswith(".pdf"):
                rewritten += ".pdf"
            if rewritten not in seen:
                seen.add(rewritten)
                pdf_urls.append(rewritten)

    # Scan all <a> tags
    pdf_kw = ["pdf", "download pdf", "full text pdf", "view pdf", "download article"]
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True).lower()
        if href.lower().endswith(".pdf") or any(kw in text for kw in pdf_kw) or "pdf" in href.lower():
            full_url = urljoin(base_url, href)
            if full_url not in seen:
                seen.add(full_url)
                pdf_urls.append(full_url)

    return pdf_urls
