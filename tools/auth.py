"""
tools/auth.py — Authentication layer for the Dataset Builder pipeline.

Provides a multi-factor gate that must pass before any pipeline run starts.
Three auth modes (configured via .env / cfg):

  MODE 1 — API-Key only  (default, lightest)
    Reads PIPELINE_API_KEY from .env.  The chatbot or batch runner
    prompts the user to enter it.  SHA-256 hash compared against
    PIPELINE_API_KEY_HASH stored in .env.

  MODE 2 — API-Key + TOTP  (recommended for shared machines)
    Same as mode 1, plus a time-based one-time password generated
    from PIPELINE_TOTP_SECRET (base32 string in .env).
    Compatible with any TOTP app (Google Authenticator, Authy, etc.).

  MODE 3 — API-Key + IP allowlist  (CI/CD / server deployments)
    Validates that the calling host IP is in PIPELINE_ALLOWED_IPS
    (comma-separated CIDR list in .env).

Configuration (.env additions)
--------------------------------
    PIPELINE_AUTH_MODE=1              # 1 | 2 | 3
    PIPELINE_API_KEY_HASH=<sha256>    # sha256(your_key)
    PIPELINE_TOTP_SECRET=BASE32SECRET # only for mode 2
    PIPELINE_ALLOWED_IPS=127.0.0.1,10.0.0.0/8  # only for mode 3
    PIPELINE_MAX_ATTEMPTS=5           # lockout after N failed attempts
    PIPELINE_LOCKOUT_SECONDS=300      # lockout duration

Usage (drop-in for chatbot.py / main.py)
-----------------------------------------
    from tools.auth import require_auth

    # Call once at startup — raises AuthError on failure
    require_auth()

Generating a key hash (run once):
    python -c "import hashlib; print(hashlib.sha256(b'your_key').hexdigest())"

Generating a TOTP secret:
    python -c "import base64, os; print(base64.b32encode(os.urandom(20)).decode())"
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import os
import socket
import struct
import time
from pathlib import Path

from tools.console_setup import console

# ── Load env if not already loaded ────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class AuthError(Exception):
    """Raised when authentication fails unrecoverably."""


# ── Config from environment ───────────────────────────────────────────────────

_AUTH_MODE         = int(os.getenv("PIPELINE_AUTH_MODE", "1"))
_KEY_HASH          = os.getenv("PIPELINE_API_KEY_HASH", "")
_TOTP_SECRET       = os.getenv("PIPELINE_TOTP_SECRET", "")
_ALLOWED_IPS_RAW   = os.getenv("PIPELINE_ALLOWED_IPS", "127.0.0.1")
_MAX_ATTEMPTS      = int(os.getenv("PIPELINE_MAX_ATTEMPTS", "5"))
_LOCKOUT_SECONDS   = int(os.getenv("PIPELINE_LOCKOUT_SECONDS", "300"))

# Lockout state (in-process; resets on restart)
_failed_attempts: int = 0
_lockout_until:   float = 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _secure_compare(a: str, b: str) -> bool:
    """Timing-safe string comparison to prevent timing attacks."""
    return hmac.compare_digest(a.encode(), b.encode())


def _totp(secret_b32: str, window: int = 1) -> set[str]:
    """
    Generate valid TOTP codes for the current window ± `window` steps (30 s each).
    Returns a set of 6-digit strings.

    Compatible with RFC 6238 / Google Authenticator.
    """
    try:
        import base64
        key = base64.b32decode(secret_b32.upper().strip())
    except Exception:
        raise AuthError(
            "PIPELINE_TOTP_SECRET is not a valid base32 string. "
            "Generate one with: python -c \"import base64,os; "
            "print(base64.b32encode(os.urandom(20)).decode())\""
        )

    codes: set[str] = set()
    now_step = int(time.time()) // 30

    for step in range(now_step - window, now_step + window + 1):
        msg = struct.pack(">Q", step)
        import hmac as _hmac, hashlib as _hl
        h = _hmac.new(key, msg, _hl.sha1).digest()
        offset = h[-1] & 0x0F
        code = struct.unpack(">I", h[offset:offset + 4])[0] & 0x7FFFFFFF
        codes.add(f"{code % 1_000_000:06d}")

    return codes


def _get_local_ip() -> str:
    """Best-effort local IP detection."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def _ip_in_allowlist(ip: str, raw_allowlist: str) -> bool:
    """Check if `ip` matches any CIDR or exact IP in the comma-separated list."""
    host = ipaddress.ip_address(ip)
    for entry in raw_allowlist.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            if "/" in entry:
                if host in ipaddress.ip_network(entry, strict=False):
                    return True
            else:
                if host == ipaddress.ip_address(entry):
                    return True
        except ValueError:
            continue
    return False


# ── Auth checks ───────────────────────────────────────────────────────────────

def _check_api_key(provided: str) -> bool:
    if not _KEY_HASH:
        console.print(
            "[yellow]⚠ PIPELINE_API_KEY_HASH not set in .env — skipping key validation.[/yellow]"
        )
        return True
    return _secure_compare(_sha256(provided), _KEY_HASH)


def _check_totp(code: str) -> bool:
    if not _TOTP_SECRET:
        console.print("[yellow]⚠ PIPELINE_TOTP_SECRET not set — skipping TOTP.[/yellow]")
        return True
    valid_codes = _totp(_TOTP_SECRET)
    return code.strip() in valid_codes


def _check_ip() -> bool:
    my_ip = _get_local_ip()
    allowed = _ip_in_allowlist(my_ip, _ALLOWED_IPS_RAW)
    if not allowed:
        console.print(f"[red]✗ IP {my_ip} not in PIPELINE_ALLOWED_IPS.[/red]")
    return allowed


# ── Lockout logic ─────────────────────────────────────────────────────────────

def _record_failure() -> None:
    global _failed_attempts, _lockout_until
    _failed_attempts += 1
    if _failed_attempts >= _MAX_ATTEMPTS:
        _lockout_until = time.time() + _LOCKOUT_SECONDS
        console.print(
            f"[bold red]🔒 Too many failed attempts. "
            f"Locked out for {_LOCKOUT_SECONDS}s.[/bold red]"
        )


def _check_lockout() -> None:
    if time.time() < _lockout_until:
        remaining = int(_lockout_until - time.time())
        raise AuthError(f"Account locked. Try again in {remaining}s.")


def _record_success() -> None:
    global _failed_attempts
    _failed_attempts = 0


# ── Prompt helpers (interactive mode) ────────────────────────────────────────

def _prompt_secret(label: str) -> str:
    """Prompt for a secret without echoing (uses getpass, falls back to input)."""
    try:
        import getpass
        return getpass.getpass(f"  {label}: ")
    except Exception:
        return input(f"  {label}: ")


# ── Public API ────────────────────────────────────────────────────────────────

def require_auth(
    api_key: str | None = None,
    totp_code: str | None = None,
    *,
    interactive: bool = True,
) -> None:
    """
    Enforce authentication before pipeline execution.

    Parameters
    ----------
    api_key : str | None
        If provided, used directly (non-interactive / batch mode).
    totp_code : str | None
        If provided, used directly for mode-2 TOTP check.
    interactive : bool
        If True (default) and credentials are missing, prompt the user.
        Set False in automated/CI contexts.

    Raises
    ------
    AuthError
        If authentication fails or the account is locked out.
    """
    _check_lockout()

    mode = _AUTH_MODE
    console.print(f"\n[bold cyan]🔐 Authentication required[/bold cyan] (mode {mode})")

    # ── Mode 3: IP allowlist (runs first — transparent to user) ──────────────
    if mode == 3:
        if not _check_ip():
            _record_failure()
            raise AuthError("IP address not authorised.")
        _record_success()
        console.print("[green]✓ IP allowlist check passed.[/green]")
        return

    # ── API key ───────────────────────────────────────────────────────────────
    if api_key is None:
        if interactive:
            api_key = _prompt_secret("Enter pipeline API key")
        else:
            raise AuthError("api_key required in non-interactive mode.")

    if not _check_api_key(api_key):
        _record_failure()
        raise AuthError("Invalid API key.")

    # ── Mode 2: TOTP ─────────────────────────────────────────────────────────
    if mode == 2:
        if totp_code is None:
            if interactive:
                totp_code = _prompt_secret("Enter TOTP code (6 digits)")
            else:
                raise AuthError("totp_code required in non-interactive mode.")

        if not _check_totp(totp_code):
            _record_failure()
            raise AuthError("Invalid TOTP code.")

    _record_success()
    console.print("[bold green]✓ Authenticated.[/bold green]\n")


def generate_key_hash(key: str) -> str:
    """
    Utility: generate the SHA-256 hash of a key.
    Print this value and store it as PIPELINE_API_KEY_HASH in .env.

    Example:
        python -c "from tools.auth import generate_key_hash; print(generate_key_hash('mysecret'))"
    """
    h = _sha256(key)
    console.print(f"SHA-256 hash: [bold]{h}[/bold]")
    console.print("Add to .env:  PIPELINE_API_KEY_HASH=" + h)
    return h


def generate_totp_secret() -> str:
    """
    Utility: generate a random TOTP secret.
    Print it, scan the QR (or enter manually) in your authenticator app,
    and store it as PIPELINE_TOTP_SECRET in .env.
    """
    import base64
    secret = base64.b32encode(os.urandom(20)).decode()
    console.print(f"TOTP secret: [bold]{secret}[/bold]")
    console.print("Add to .env: PIPELINE_TOTP_SECRET=" + secret)
    console.print(
        "Then open Google Authenticator → Add account → Enter secret manually."
    )
    return secret


# ── CLI convenience ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2 and sys.argv[1] == "hash":
        key = input("Enter key to hash: ")
        generate_key_hash(key)
    elif len(sys.argv) == 2 and sys.argv[1] == "totp":
        generate_totp_secret()
    else:
        print("Usage:")
        print("  python -m tools.auth hash   # generate API key hash")
        print("  python -m tools.auth totp   # generate TOTP secret")
