"""
tools/browser.py — Maximum-strength anti-bot bypass toolkit.

Techniques used (state of the art as of 2025):
  1. curl_cffi  — Chrome/Firefox TLS fingerprint impersonation (JA3/JA4)
  2. Playwright — Full browser with STEALTH_JS (webdriver flag removal,
                  canvas noise, WebGL spoofing, plugin faking)
  3. Bezier-curve mouse movement — bypasses ML-based mouse tracking detectors
  4. Per-domain rate throttling — avoids triggering velocity-based blocks
  5. Proxy pool rotation — different exit IPs per domain to avoid IP bans
  6. HTTP/2 fingerprint matching — curl_cffi matches browser H2 settings
  7. Accept-CH header spoofing — Chrome client hints
  8. Realistic session cookies — cookies persist per domain within a session
  9. Exponential backoff with jitter on 429/503
 10. Cloudflare / hCaptcha detection and Playwright fallback
"""

from __future__ import annotations

import collections
import hashlib
import math
import os
import random
import re
import time
from pathlib import Path
from urllib.parse import urlparse

from tools.console_setup import console
import config as cfg

# ── curl_cffi ─────────────────────────────────────────────────────────────────
try:
    from curl_cffi import requests as cffi_requests
    CURL_AVAILABLE = True
except ImportError:
    import requests as cffi_requests
    CURL_AVAILABLE = False

import requests as _std_requests

# ── Proxy pool (round-robin) ──────────────────────────────────────────────────
_proxy_pool: list[str] = list(cfg.PROXY_LIST)
_proxy_idx: int = 0


def _next_proxy() -> dict | None:
    global _proxy_idx
    if not _proxy_pool:
        return None
    p = _proxy_pool[_proxy_idx % len(_proxy_pool)]
    _proxy_idx += 1
    return {"http": p, "https": p}


# ── Per-domain rate throttle ──────────────────────────────────────────────────
_domain_last_hit: dict[str, float] = collections.defaultdict(float)
_domain_min_gap = 3.0   # seconds between requests to same domain
_domain_max_gap = 8.0


def _throttle_domain(url: str) -> None:
    domain = urlparse(url).netloc.lower()
    now    = time.time()
    gap    = _domain_last_hit[domain]
    wait   = _domain_min_gap - (now - gap)
    if wait > 0:
        jitter = random.uniform(0, _domain_max_gap - _domain_min_gap)
        time.sleep(wait + jitter)
    _domain_last_hit[domain] = time.time()


# ── Session cookie jar per domain ─────────────────────────────────────────────
_session_cookies: dict[str, dict] = collections.defaultdict(dict)


# ── User agents ───────────────────────────────────────────────────────────────
_USER_AGENTS = [
    # Chrome Windows (latest 3 versions)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    # Chrome Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    # Chrome Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    # Firefox
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.5; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0",
    # Edge
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
    # Safari
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    # Mobile
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
]

# Matching sec-ch-ua hints for Chrome versions
_SEC_CH_UA = {
    "131": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "130": '"Google Chrome";v="130", "Chromium";v="130", "Not_A Brand";v="24"',
    "129": '"Google Chrome";v="129", "Chromium";v="129", "Not_A Brand";v="24"',
}

_REFERERS = [
    "https://www.google.com/",
    "https://scholar.google.com/",
    "https://www.bing.com/",
    "https://duckduckgo.com/",
    "https://pubmed.ncbi.nlm.nih.gov/",
    "",
]

_ACCEPT_LANGS = [
    "en-US,en;q=0.9",
    "en-US,en;q=0.9,es;q=0.8",
    "en-GB,en;q=0.9,en-US;q=0.8",
    "en-US,en;q=0.5",
]

_CURL_IMPERSONATIONS = [
    "chrome131", "chrome130", "chrome129",
    "firefox133", "firefox132",
    "safari17_4",
    "edge131",
]


def random_ua() -> str:
    return random.choice(_USER_AGENTS)


def _build_headers(url: str = "", custom_headers: dict | None = None) -> dict:
    """Build maximally realistic browser headers with client hints."""
    ua = random_ua()
    # Detect Chrome version for matching sec-ch-ua
    chrome_ver_m = re.search(r'Chrome/(\d+)', ua)
    chrome_ver   = chrome_ver_m.group(1) if chrome_ver_m else "131"

    h = {
        "User-Agent":                ua,
        "Accept":                    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language":           random.choice(_ACCEPT_LANGS),
        "Accept-Encoding":           "gzip, deflate, br, zstd",
        "Connection":                "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest":            "document",
        "Sec-Fetch-Mode":            "navigate",
        "Sec-Fetch-Site":            "none",
        "Sec-Fetch-User":            "?1",
        "Cache-Control":             "max-age=0",
        "DNT":                       "1",
        "Priority":                  "u=0, i",
    }

    # Chrome-specific client hints
    if "Chrome" in ua and "Edg/" not in ua:
        h["sec-ch-ua"]          = _SEC_CH_UA.get(chrome_ver,
            f'"Google Chrome";v="{chrome_ver}", "Chromium";v="{chrome_ver}", "Not_A Brand";v="24"')
        h["sec-ch-ua-mobile"]   = "?0"
        h["sec-ch-ua-platform"] = '"Windows"' if "Windows" in ua else '"macOS"'

    # Referer
    ref = random.choice(_REFERERS)
    if ref:
        h["Referer"]         = ref
        h["Sec-Fetch-Site"]  = "cross-site"

    if custom_headers and url:
        domain = urlparse(url).netloc.lower()
        for pattern, headers in custom_headers.items():
            if pattern.lower() in domain:
                h.update(headers)
                break

    return h


# ── Stealth JS — comprehensive bot signal removal ─────────────────────────────
STEALTH_JS = """
// 1. Remove webdriver flag
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
delete navigator.__proto__.webdriver;

// 2. Chrome runtime
window.chrome = {
    runtime: { id: undefined, connect: () => {}, sendMessage: () => {} },
    loadTimes: function() { return { firstPaintTime: Math.random() * 0.5 }; },
    csi: function() { return { startE: Date.now() }; },
    app: { isInstalled: false }
};

// 3. Permissions — avoid automation detection
const _origQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (parameters) =>
    parameters.name === 'notifications'
        ? Promise.resolve({ state: Notification.permission })
        : _origQuery(parameters);

// 4. Realistic plugins
Object.defineProperty(navigator, 'plugins', {
    get: () => {
        const plugins = [
            { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer', description: 'Portable Document Format', length: 1 },
            { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', description: '', length: 1 },
            { name: 'Native Client', filename: 'internal-nacl-plugin', description: '', length: 2 },
        ];
        plugins.refresh = () => {};
        return plugins;
    }
});

Object.defineProperty(navigator, 'mimeTypes', {
    get: () => ({ length: 4, 0: { type: 'application/pdf' } })
});

// 5. Languages, platform, hardware
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
Object.defineProperty(navigator, 'language',  { get: () => 'en-US' });
Object.defineProperty(navigator, 'platform',  { get: () => 'Win32' });
Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });
Object.defineProperty(navigator, 'deviceMemory', { get: () => 8 });

// 6. WebGL — realistic vendor/renderer
const _getParam = WebGLRenderingContext.prototype.getParameter;
WebGLRenderingContext.prototype.getParameter = function(parameter) {
    if (parameter === 37445) return 'Intel Inc.';
    if (parameter === 37446) return 'Intel Iris OpenGL Engine';
    return _getParam.call(this, parameter);
};

// 7. Canvas fingerprint noise (subtle)
const _toDataURL = HTMLCanvasElement.prototype.toDataURL;
HTMLCanvasElement.prototype.toDataURL = function(type) {
    if (type === 'image/png' && this.width > 16) {
        const ctx = this.getContext('2d');
        if (ctx) {
            const n = (Math.random() - 0.5) * 0.02;
            ctx.fillStyle = `rgba(255,255,255,${Math.abs(n)})`;
            ctx.fillRect(0, 0, 1, 1);
        }
    }
    return _toDataURL.apply(this, arguments);
};

const _getImageData = CanvasRenderingContext2D.prototype.getImageData;
CanvasRenderingContext2D.prototype.getImageData = function(x, y, w, h) {
    const data = _getImageData.call(this, x, y, w, h);
    data.data[0] ^= 1;
    return data;
};

// 8. Connection info
Object.defineProperty(navigator, 'connection', {
    get: () => ({
        effectiveType: '4g', rtt: 50 + Math.random() * 20,
        downlink: 10 + Math.random() * 5, saveData: false
    })
});

// 9. Screen — realistic values
Object.defineProperty(screen, 'colorDepth', { get: () => 24 });
Object.defineProperty(screen, 'pixelDepth', { get: () => 24 });

// 10. Remove automation-related properties
['__nightmare', '__selenium', '__webdriverFunc', '_phantom',
 'callPhantom', '_Selenium_IDE_Recorder', '__driver_evaluate',
 '__webdriver_evaluate', '__selenium_evaluate', '__fxdriver_evaluate',
 '__driver_unwrapped', '__webdriver_unwrapped', '__selenium_unwrapped',
 '__fxdriver_unwrapped'].forEach(prop => {
    if (prop in window) delete window[prop];
});

// 11. Realistic history length
Object.defineProperty(history, 'length', { get: () => Math.floor(Math.random() * 8) + 2 });

// 12. Audio context fingerprint noise
if (window.AudioContext || window.webkitAudioContext) {
    const AC = window.AudioContext || window.webkitAudioContext;
    const origCreateAnalyser = AC.prototype.createAnalyser;
    AC.prototype.createAnalyser = function() {
        const analyser = origCreateAnalyser.call(this);
        const origGetFloat = analyser.getFloatFrequencyData.bind(analyser);
        analyser.getFloatFrequencyData = function(arr) {
            origGetFloat(arr);
            for (let i = 0; i < arr.length; i++) arr[i] += (Math.random() - 0.5) * 0.02;
        };
        return analyser;
    };
}
"""

# ── Bot detection signals ─────────────────────────────────────────────────────
_CF_SIGNALS = [
    "cloudflare", "cf-browser-verification", "ray id",
    "just a moment", "checking your browser", "enable javascript",
    "attention required", "cf-challenge", "turnstile",
    "please verify you are a human", "__cf_chl",
]

_CAPTCHA_SIGNALS = [
    "recaptcha", "hcaptcha", "captcha", "g-recaptcha",
    "cf-turnstile", "verify you are human", "i'm not a robot",
    "bot detection", "please complete the security check",
    "security check", "prove you are human",
]

_PAYWALL_SIGNALS = [
    "subscribe to read", "purchase access", "buy this article",
    "log in to read", "create account to continue",
    "register to view", "institutional access required",
]


def _detect_block_reason(html: str, status_code: int = 200) -> str | None:
    if not html:
        return None
    snippet = html[:6000].lower()
    if any(sig in snippet for sig in _CAPTCHA_SIGNALS):
        return "captcha"
    if any(sig in snippet for sig in _CF_SIGNALS):
        return "cloudflare"
    if status_code == 403:
        return "403"
    if status_code == 429:
        return "429_exhausted"
    return None


# ── Browser management ────────────────────────────────────────────────────────
_playwright = None
_browser    = None


def init_browser():
    global _playwright, _browser
    if _browser is not None:
        return _browser
    try:
        from playwright.sync_api import sync_playwright
        _playwright = sync_playwright().start()
        w = random.randint(1366, 1920)
        h = random.randint(768, 1080)
        _browser = _playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-infobars",
                "--disable-background-networking",
                "--disable-default-apps",
                "--disable-extensions",
                "--disable-gpu",
                "--disable-sync",
                "--disable-translate",
                "--metrics-recording-only",
                "--mute-audio",
                f"--window-size={w},{h}",
                "--lang=en-US",
                "--ignore-certificate-errors",
            ],
        )
        console.print("[green]✓ Browser launched (stealth mode)[/green]")
        return _browser
    except Exception as e:
        console.print(f"[yellow]⚠ Browser unavailable: {e}[/yellow]")
        return None


def close_browser():
    global _playwright, _browser
    if _browser:
        try: _browser.close()
        except Exception: pass
        _browser = None
    if _playwright:
        try: _playwright.stop()
        except Exception: pass
        _playwright = None


# ── Bezier mouse movement ─────────────────────────────────────────────────────

def _bezier_point(t: float, p0: tuple, p1: tuple,
                   p2: tuple, p3: tuple) -> tuple:
    u = 1 - t
    return (
        u**3 * p0[0] + 3*u**2*t*p1[0] + 3*u*t**2*p2[0] + t**3*p3[0],
        u**3 * p0[1] + 3*u**2*t*p1[1] + 3*u*t**2*p2[1] + t**3*p3[1],
    )


def _bezier_mouse_move(page, start: tuple, end: tuple, steps: int = 20):
    """Move mouse along a realistic Bezier curve — defeats ML mouse trackers."""
    cx1 = start[0] + random.randint(-120, 120)
    cy1 = start[1] + random.randint(-60, 60)
    cx2 = end[0]   + random.randint(-120, 120)
    cy2 = end[1]   + random.randint(-60, 60)
    for i in range(steps + 1):
        t = i / steps
        x, y = _bezier_point(t, start, (cx1, cy1), (cx2, cy2), end)
        x += random.gauss(0, 0.8)
        y += random.gauss(0, 0.8)
        page.mouse.move(x, y)
        pause = 0.004 + 0.025 * math.sin(t * math.pi)
        time.sleep(pause + random.uniform(0, 0.008))


def simulate_human(page):
    """Simulate realistic human interaction: Bezier mouse + reading scroll."""
    try:
        vw = page.viewport_size
        if not vw:
            return
        w, h = vw["width"], vw["height"]
        cx, cy = w // 2, h // 2

        # Bezier mouse movements to random spots
        current = (cx, cy)
        for _ in range(random.randint(2, 5)):
            target = (random.randint(80, w-80), random.randint(80, h-80))
            _bezier_mouse_move(page, current, target,
                               steps=random.randint(15, 30))
            current = target
            time.sleep(random.uniform(0.08, 0.35))

        # Reading scroll: variable-speed downward scroll
        total_scroll = random.randint(400, 1500)
        scrolled = 0
        while scrolled < total_scroll:
            chunk = random.randint(60, 280)
            page.mouse.wheel(0, chunk)
            scrolled += chunk
            # Longer pause after bigger chunks (simulates reading)
            time.sleep(random.uniform(0.25, 1.0 + chunk / 400))

        # Occasional scroll-back (re-reading behaviour)
        if random.random() < 0.35:
            page.mouse.wheel(0, -random.randint(60, 250))
            time.sleep(random.uniform(0.3, 0.9))

        # Final mouse move
        _bezier_mouse_move(page, current,
                            (random.randint(80, w-80), random.randint(80, h-80)),
                            steps=10)
        time.sleep(random.uniform(0.2, 0.6))

    except Exception:
        pass


# ── Exponential backoff fetch ─────────────────────────────────────────────────

def _exponential_backoff_fetch(
    url: str,
    headers: dict | None = None,
    max_retries: int | None = None,
    base_delay: float | None = None,
    timeout: int = 25,
) -> object | None:
    if max_retries  is None: max_retries  = cfg.MAX_BACKOFF_RETRIES
    if base_delay   is None: base_delay   = cfg.BACKOFF_BASE_SECONDS
    if headers      is None: headers      = _build_headers(url)

    proxy = _next_proxy()

    for attempt in range(max_retries + 1):
        try:
            if CURL_AVAILABLE:
                impersonation = random.choice(_CURL_IMPERSONATIONS)
                resp = cffi_requests.get(
                    url,
                    headers=headers,
                    timeout=timeout,
                    impersonate=impersonation,
                    allow_redirects=True,
                    proxies=proxy,
                )
            else:
                resp = _std_requests.get(
                    url, headers=headers, timeout=timeout,
                    allow_redirects=True, proxies=proxy,
                )

            if resp.status_code < 400:
                return resp

            if resp.status_code in (429, 503, 520, 521, 522, 523, 524):
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt) + random.uniform(0.5, 2.0)
                    console.print(
                        f"    [yellow]⏳ HTTP {resp.status_code} — "
                        f"backoff {delay:.1f}s (attempt {attempt+1}/{max_retries})[/yellow]"
                    )
                    time.sleep(delay)
                    # Rotate proxy on rate-limit
                    proxy = _next_proxy()
                    continue
                return resp

            if resp.status_code in (403, 406):
                body = resp.text[:2000].lower() if hasattr(resp, "text") else ""
                if any(sig in body for sig in ["cloudflare", "ray id", "turnstile"]):
                    console.print(f"    [yellow]🛡 Cloudflare detected[/yellow]")
                return resp

            return resp

        except Exception as e:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0.5, 2.0)
                console.print(f"    [yellow]⚠ Request error: {e} — retry {delay:.1f}s[/yellow]")
                time.sleep(delay)
                proxy = _next_proxy()
            else:
                console.print(f"    [red]✗ Failed after {max_retries} retries: {e}[/red]")
                return None
    return None


# ── FetchResult ───────────────────────────────────────────────────────────────

class FetchResult:
    def __init__(self):
        self.html:         str  = ""
        self.status_code:  int  = 0
        self.method:       str  = "none"
        self.blocked:      bool = False
        self.block_reason: str  = ""
        self.success:      bool = False


# ── smart_fetch — unified entry point ─────────────────────────────────────────

def smart_fetch(url: str, custom_headers: dict | None = None,
                paywall_check: bool = True) -> FetchResult:
    """
    Fetch with graceful degradation:
      1. curl_cffi with random Chrome/Firefox TLS impersonation + proxy
      2. Playwright with full STEALTH_JS + Bezier mouse simulation
      3. Standard requests (last resort)
    """
    result  = FetchResult()
    headers = _build_headers(url, custom_headers)

    # Per-domain throttle
    _throttle_domain(url)

    # ── Step 1: curl_cffi ────────────────────────────────────────────────
    resp = _exponential_backoff_fetch(url, headers=headers)

    if resp is not None:
        result.status_code = resp.status_code
        if resp.status_code < 400:
            html  = resp.text if hasattr(resp, "text") else ""
            block = _detect_block_reason(html, resp.status_code)
            if not block:
                result.html    = html
                result.method  = "curl_cffi" if CURL_AVAILABLE else "requests"
                result.success = True
                return result
            console.print(f"    [yellow]🛡 Block ({block}) — trying Playwright[/yellow]")
            result.block_reason = block
        elif resp.status_code in (403, 429, 503):
            body  = resp.text if hasattr(resp, "text") else ""
            block = _detect_block_reason(body, resp.status_code)
            result.block_reason = block or str(resp.status_code)
            console.print(f"    [yellow]HTTP {resp.status_code} — trying Playwright[/yellow]")

    # ── Step 2: Playwright ───────────────────────────────────────────────
    browser = init_browser()
    if browser:
        try:
            ua      = random_ua()
            context = browser.new_context(
                user_agent=ua,
                viewport={"width":  random.randint(1366, 1920),
                           "height": random.randint(768, 1080)},
                locale="en-US",
                timezone_id="America/New_York",
                color_scheme="light",
                java_script_enabled=True,
                bypass_csp=True,
                extra_http_headers={
                    "Accept-Language": random.choice(_ACCEPT_LANGS),
                    "DNT": "1",
                },
            )
            page = context.new_page()
            try:
                page.add_init_script(STEALTH_JS)
                page.goto(url, wait_until="domcontentloaded", timeout=40_000)
                simulate_human(page)
                # Wait for dynamic content
                try:
                    page.wait_for_load_state("networkidle", timeout=12_000)
                except Exception:
                    pass
                time.sleep(random.uniform(1.2, 2.5))
                html  = page.content()
                block = _detect_block_reason(html)
                if block:
                    result.blocked      = True
                    result.block_reason = block
                    result.html         = html
                    result.method       = "playwright"
                    result.status_code  = 200
                    console.print(f"    [red]✗ Playwright also blocked ({block})[/red]")
                else:
                    result.html        = html
                    result.method      = "playwright"
                    result.success     = True
                    result.status_code = 200
            except Exception as e:
                console.print(f"    [yellow]⚠ Playwright page error: {e}[/yellow]")
                result.block_reason = "timeout" if "timeout" in str(e).lower() else "error"
            finally:
                try: context.close()
                except Exception: pass
        except Exception as e:
            console.print(f"    [yellow]⚠ Playwright context failed: {e}[/yellow]")

    # ── Step 3: Standard requests last resort ────────────────────────────
    if not result.success and not result.html:
        try:
            resp = _std_requests.get(url, headers=headers, timeout=20,
                                      allow_redirects=True)
            if resp.status_code < 400:
                result.html        = resp.text
                result.method      = "requests_fallback"
                result.success     = True
                result.status_code = resp.status_code
            else:
                result.status_code = resp.status_code
        except Exception:
            pass

    return result


# ── Paywall management ────────────────────────────────────────────────────────
_paywall_patterns: list[str] = []
_paywall_signals_html: list[str] = []


def configure_paywall(domains: list[str], signals: list[str]) -> None:
    global _paywall_patterns, _paywall_signals_html
    _paywall_patterns      = [str(d).lower() for d in domains]
    _paywall_signals_html  = [str(s).lower() for s in signals]


def is_paywalled(url: str) -> bool:
    if not _paywall_patterns:
        return False
    u = url.lower()
    return any(p in u for p in _paywall_patterns)


def html_is_paywalled(html: str) -> bool:
    signals = _paywall_signals_html or _PAYWALL_SIGNALS
    snippet = html[:8000].lower()
    return sum(1 for sig in signals if sig in snippet) >= 2


# ── Legacy API (kept for compatibility) ───────────────────────────────────────

def scrape_with_playwright(url: str) -> str:
    return smart_fetch(url).html


def scrape_with_requests(url: str) -> str:
    headers = _build_headers(url)
    resp    = _exponential_backoff_fetch(url, headers=headers)
    if resp and resp.status_code < 400:
        return resp.text if hasattr(resp, "text") else ""
    return ""


def download_pdf(url: str, page=None) -> str | None:
    """Download a PDF to the temp directory. Returns local path or None."""
    os.makedirs(cfg.PDF_TEMP_DIR, exist_ok=True)
    try:
        base     = url.split("?")[0].rstrip("/")
        filename = base.split("/")[-1]
        if not filename or filename.lower() in ("pdf", ""):
            filename = hashlib.md5(url.encode()).hexdigest()[:10]
        import re as _re
        filename = _re.sub(r"[^\w\-.]", "_", filename)
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"
        filepath = os.path.join(cfg.PDF_TEMP_DIR, filename)

        if page:
            resp = page.request.get(url, timeout=30_000)
            if not resp.ok:
                return None
            body = resp.body()
            ct   = resp.headers.get("content-type", "").lower()
            if "html" in ct or b"<html" in body[:512].lower():
                try:
                    with page.expect_download(timeout=20_000) as dl_info:
                        page.goto(url, wait_until="commit")
                    dl_info.value.save_as(filepath)
                    return filepath
                except Exception:
                    return None
            with open(filepath, "wb") as fh:
                fh.write(body)
        else:
            headers = _build_headers(url)
            resp    = _exponential_backoff_fetch(url, headers=headers)
            if not resp or resp.status_code >= 400:
                return None
            ct      = getattr(resp, "headers", {}).get("content-type", "").lower()
            content = resp.content if hasattr(resp, "content") else b""
            if b"<html" in content[:512].lower():
                return None
            if "pdf" not in ct and not url.lower().endswith(".pdf"):
                return None
            with open(filepath, "wb") as fh:
                fh.write(content)

        console.print(f"  [dim]📥 PDF downloaded: {filename}[/dim]")
        return filepath
    except Exception as e:
        console.print(f"  [yellow]⚠ PDF download failed: {e}[/yellow]")
        return None
