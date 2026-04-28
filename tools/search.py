"""
tools/search.py — Web search with dynamic topic-aware domain configuration.
"""

import random
from tools.console_setup import console

import config as cfg



# ── Dynamic domain config ─────────────────────────────────────────────────────
_relevant_domains: list[str] = []
_blocked_domains: list[str] = []
_priority_tiers: list[list[str]] = []

JUNK_URL_PATTERNS = [
    "/tag/", "/category/", "/blog/", "/news/", "/about/",
    "/contact/", "/login", "/signup", "/cart", "/product/", "/shop/",
    "/search?", "/topic/", "/author/", "/press-release/",
]


def configure_domains(domain_config: dict) -> None:
    global _relevant_domains, _blocked_domains, _priority_tiers
    _relevant_domains = [str(d).lower() for d in domain_config.get("relevant_domains", [])]
    _blocked_domains = [str(d).lower() for d in domain_config.get("blocked_domains", [])]
    raw_tiers = domain_config.get("priority_tiers", [])
    _priority_tiers = [
        [str(d).lower() for d in tier] for tier in raw_tiers if isinstance(tier, list)
    ]
    console.print(
        f"  [dim]Domain config: {len(_relevant_domains)} relevant, "
        f"{len(_blocked_domains)} blocked, {len(_priority_tiers)} tiers[/dim]"
    )


def is_blocked(url: str) -> bool:
    u = url.lower()
    if any(domain in u for domain in _blocked_domains):
        return True
    if any(pattern in u for pattern in JUNK_URL_PATTERNS):
        return True
    return False


def url_priority(url: str) -> int:
    u = url.lower()
    for tier_idx, tier_domains in enumerate(_priority_tiers):
        if any(d in u for d in tier_domains):
            return tier_idx
    if any(d in u for d in _relevant_domains):
        return len(_priority_tiers) + 1
    return 99


def search_duckduckgo(queries: list[str], limit: int = 10) -> list[dict]:
    """
    Search DuckDuckGo for sources.
    Returns list of {title, url, snippet, priority, is_relevant}
    """
    results: list[dict] = []
    seen_urls: set[str] = set()
    per_query = max(limit // len(queries) + 2, 2)
    buffer_cap = limit * 2

    for q in queries:
        if len(results) >= buffer_cap:
            break
        try:
            console.print(f"  [dim]🔍 {q[:90]}[/dim]")
            from ddgs import DDGS
            with DDGS() as ddgs:
                hits = list(ddgs.text(q, max_results=per_query))

            for hit in hits:
                url = hit.get("href", "")
                if not url or url in seen_urls:
                    continue
                if is_blocked(url):
                    console.print(f"  [dim]✗ Blocked: {url[:70]}[/dim]")
                    continue
                seen_urls.add(url)
                results.append({
                    "title":       hit.get("title", ""),
                    "url":         url,
                    "snippet":     hit.get("body", ""),
                    "is_relevant": any(d in url.lower() for d in _relevant_domains),
                    "priority":    url_priority(url),
                })

        except Exception as e:
            console.print(f"  [yellow]⚠ Search failed: {e}[/yellow]")

    results.sort(key=lambda r: r["priority"])
    results = results[:limit]

    if results:
        relevant = sum(1 for r in results if r["is_relevant"])
        console.print(f"  [green]✓ {len(results)} source(s) — {relevant} from relevant domains[/green]")
    else:
        console.print("[yellow]⚠ No sources found.[/yellow]")

    return results
