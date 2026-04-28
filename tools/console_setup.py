"""
tools/console_setup.py — Shared Rich console with Windows UTF-8 fix.

All modules should import console from here instead of creating their own.
"""

import sys
import io

# Force UTF-8 on Windows to prevent cp1252 encoding crashes with emoji/unicode
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console

console = Console(force_terminal=True)
