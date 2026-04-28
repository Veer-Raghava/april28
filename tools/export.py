"""
tools/export.py — Fancy CLI output: Rich panels, dashboards, color-coded tables,
                  blocked source display, quality reporting.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TaskProgressColumn
from rich import box
from tools.console_setup import console

# ── Color scheme ──────────────────────────────────────────────────────────────
COLORS = {
    "brand":     "bold cyan",
    "success":   "bold green",
    "warning":   "bold yellow",
    "error":     "bold red",
    "dim":       "dim white",
    "accent":    "bold magenta",
    "highlight": "bold bright_white",
}


# ── Save functions ────────────────────────────────────────────────────────────

def save_csv(rows: list[dict], columns: list[str], output_path: str) -> str:
    """Save rows to CSV. Returns actual path used."""
    if not rows:
        return output_path

    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = "N/A"

    extra_cols = [c for c in df.columns if c not in columns]
    df = df[columns + extra_cols]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    except PermissionError:
        stem, ext = os.path.splitext(output_path)
        output_path = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        console.print(f"[yellow]⚠ File locked — saving as {os.path.basename(output_path)}[/yellow]")
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

    console.print(f"  [green]💾 Saved {len(df)} rows → {output_path}[/green]")
    return output_path


def save_rejected(rows: list[dict], output_dir: str) -> str | None:
    """Save rejected rows to a separate CSV."""
    if not rows:
        return None
    path = os.path.join(output_dir, "rejected.csv")
    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    console.print(f"  [dim]Rejected rows → {path}[/dim]")
    return path


def save_json(rows: list[dict], output_path: str) -> None:
    """Save rows as JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    json_path = output_path.replace(".csv", ".json")
    Path(json_path).write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    console.print(f"  [dim]JSON → {json_path}[/dim]")


# ── Fancy Data Preview ────────────────────────────────────────────────────────

def display_preview(rows: list[dict], columns: list[str], max_rows: int = 10) -> None:
    """Display a fancy Rich table preview with color-coded confidence."""
    if not rows:
        return

    try:
        df = pd.DataFrame(rows)
        preview_cols = [c for c in columns if c not in ("_issues",)][:8]

        table = Table(
            title=f"[bold cyan]📊 Results Preview[/bold cyan] — {len(df)} rows total",
            show_lines=True,
            expand=False,
            border_style="bright_blue",
            title_style="bold cyan",
            header_style="bold bright_white on dark_blue",
            box=box.DOUBLE_EDGE,
            padding=(0, 1),
        )

        for col in preview_cols:
            style = "cyan" if col in ("Source_URL",) else "white"
            table.add_column(str(col), max_width=38, overflow="ellipsis", style=style)

        for _, row in df.head(max_rows).iterrows():
            cells = []
            for col in preview_cols:
                val = str(row.get(col, ""))[:38]
                if not val or val.lower() in ("n/a", "none", ""):
                    cells.append("[red dim]N/A[/red dim]")
                elif col == "row_confidence":
                    try:
                        conf = float(val)
                        color = "green" if conf >= 0.7 else "yellow" if conf >= 0.4 else "red"
                        cells.append(f"[{color}]{conf:.2f}[/{color}]")
                    except ValueError:
                        cells.append(val)
                else:
                    cells.append(f"[white]{val}[/white]")
            table.add_row(*cells)

        if len(df) > max_rows:
            table.add_row(*[f"[dim]… +{len(df) - max_rows} more[/dim]"] + [""] * (len(preview_cols) - 1))

        console.print()
        console.print(table)
    except Exception as e:
        console.print(f"[yellow]⚠ Preview failed: {e}[/yellow]")


# ── Quality Report ────────────────────────────────────────────────────────────

def display_quality_report(
    accepted: int,
    rejected: int,
    null_rates: dict[str, float],
    issues: dict[str, int],
) -> None:
    """Display a fancy quality summary panel."""
    lines: list[str] = []

    # Header stats
    total = accepted + rejected
    acc_pct = accepted / max(total, 1) * 100
    lines.append(f"  [green]✓ Accepted:[/green] {accepted:,} rows ({acc_pct:.0f}%)")
    lines.append(f"  [red]✗ Rejected:[/red] {rejected:,} rows ({100 - acc_pct:.0f}%)")
    lines.append("")

    # Null rates with visual bars
    if null_rates:
        lines.append("  [bold]Per-column completeness:[/bold]")
        max_name_len = max(len(c) for c in null_rates) if null_rates else 10
        for col, rate in sorted(null_rates.items(), key=lambda x: -x[1]):
            fill_pct = 1.0 - rate
            bar_len = 20
            filled = int(fill_pct * bar_len)
            empty = bar_len - filled

            color = "green" if rate < 0.2 else "yellow" if rate < 0.5 else "red"
            bar = f"[{color}]{'█' * filled}{'░' * empty}[/{color}]"
            lines.append(f"    {col:<{max_name_len}}  {bar} [{color}]{fill_pct:>5.0%}[/{color}]")
        lines.append("")

    # Issues
    if issues:
        lines.append("  [bold]Issues detected:[/bold]")
        for issue, count in sorted(issues.items(), key=lambda x: -x[1]):
            icon = "⚠" if count < 5 else "🔴"
            lines.append(f"    {icon} [yellow]{issue}:[/yellow] {count}")

    panel = Panel(
        "\n".join(lines),
        title="[bold bright_white]📊 Quality Report[/bold bright_white]",
        border_style="bright_blue",
        padding=(1, 2),
        expand=False,
    )
    console.print()
    console.print(panel)


# ── Blocked Sources Display ───────────────────────────────────────────────────

def display_blocked_sources(blocked_sources: list) -> None:
    """Display blocked/paywalled sources in a fancy table."""
    if not blocked_sources:
        return

    table = Table(
        title=f"[bold red]🛡 Blocked Sources[/bold red] — {len(blocked_sources)} source(s)",
        show_lines=True,
        expand=False,
        border_style="red",
        title_style="bold red",
        header_style="bold bright_white on dark_red",
        box=box.ROUNDED,
        padding=(0, 1),
    )

    table.add_column("#", style="dim", width=3)
    table.add_column("URL", max_width=55, overflow="ellipsis", style="white")
    table.add_column("Reason", max_width=18, style="yellow")
    table.add_column("HTTP", width=5, style="dim")
    table.add_column("Details", max_width=30, overflow="ellipsis", style="dim")

    reason_icons = {
        "paywall": "💰",
        "cloudflare": "☁️",
        "captcha": "🤖",
        "403": "🚫",
        "429_exhausted": "⏳",
        "timeout": "⏰",
        "bot_detect": "🛡",
    }

    for i, bs in enumerate(blocked_sources, 1):
        url = bs.url if hasattr(bs, 'url') else str(bs.get('url', ''))
        reason = bs.reason if hasattr(bs, 'reason') else str(bs.get('reason', ''))
        status = bs.status_code if hasattr(bs, 'status_code') else bs.get('status_code', 0)
        details = bs.details if hasattr(bs, 'details') else str(bs.get('details', ''))

        icon = reason_icons.get(reason, "❓")
        table.add_row(
            str(i),
            url[:55],
            f"{icon} {reason}",
            str(status) if status else "—",
            details[:30],
        )

    console.print()
    console.print(table)


# ── Rejected Rows Display ────────────────────────────────────────────────────

def display_rejected_rows(rejected_rows: list[dict], columns: list[str], max_rows: int = 8) -> None:
    """Display rejected data rows with their issues."""
    if not rejected_rows:
        return

    preview_cols = [c for c in columns if c not in ("_issues", "row_confidence")][:6]
    preview_cols.append("row_confidence")
    if any("_issues" in r for r in rejected_rows):
        preview_cols.append("_issues")

    table = Table(
        title=f"[bold yellow]❌ Rejected Rows[/bold yellow] — {len(rejected_rows)} row(s)",
        show_lines=True,
        expand=False,
        border_style="yellow",
        title_style="bold yellow",
        header_style="bold bright_white on dark_orange3",
        box=box.ROUNDED,
        padding=(0, 1),
    )

    for col in preview_cols:
        table.add_column(col, max_width=30, overflow="ellipsis")

    for row in rejected_rows[:max_rows]:
        cells = []
        for col in preview_cols:
            val = str(row.get(col, ""))[:30]
            if col == "row_confidence":
                try:
                    conf = float(val)
                    cells.append(f"[red]{conf:.2f}[/red]")
                except ValueError:
                    cells.append(val)
            elif col == "_issues":
                cells.append(f"[yellow]{val}[/yellow]")
            elif not val or val.lower() in ("n/a", "none"):
                cells.append("[red dim]N/A[/red dim]")
            else:
                cells.append(val)
        table.add_row(*cells)

    if len(rejected_rows) > max_rows:
        table.add_row(*[f"[dim]… +{len(rejected_rows) - max_rows} more[/dim]"] + [""] * (len(preview_cols) - 1))

    console.print()
    console.print(table)


# ── Pipeline Dashboard ────────────────────────────────────────────────────────

def display_pipeline_dashboard(
    sources_total: int,
    sources_done: int,
    rows_extracted: int,
    blocked_count: int,
    elapsed: float,
    iteration: int,
    max_iterations: int,
) -> None:
    """Display a compact pipeline status dashboard."""
    # Build status cards
    cards = []

    # Sources card
    src_pct = sources_done / max(sources_total, 1) * 100
    src_bar_len = 15
    src_filled = int(src_pct / 100 * src_bar_len)
    src_bar = f"[green]{'█' * src_filled}[/green][dim]{'░' * (src_bar_len - src_filled)}[/dim]"
    cards.append(Panel(
        f"[bold]{sources_done}[/bold] / {sources_total}\n{src_bar} {src_pct:.0f}%",
        title="[cyan]📡 Sources[/cyan]", border_style="cyan", width=24, padding=(0, 1),
    ))

    # Rows card
    cards.append(Panel(
        f"[bold green]{rows_extracted}[/bold green] rows\n[dim]extracted so far[/dim]",
        title="[green]📊 Rows[/green]", border_style="green", width=24, padding=(0, 1),
    ))

    # Blocked card
    block_color = "red" if blocked_count > 3 else "yellow" if blocked_count > 0 else "green"
    cards.append(Panel(
        f"[bold {block_color}]{blocked_count}[/bold {block_color}] blocked\n[dim]paywall/bot/captcha[/dim]",
        title="[red]🛡 Blocked[/red]", border_style=block_color, width=24, padding=(0, 1),
    ))

    # Time card
    m, s = divmod(int(elapsed), 60)
    cards.append(Panel(
        f"[bold]{m}:{s:02d}[/bold]\n[dim]iter {iteration}/{max_iterations}[/dim]",
        title="[magenta]⏱ Time[/magenta]", border_style="magenta", width=24, padding=(0, 1),
    ))

    console.print()
    console.print(Columns(cards, expand=False, padding=(0, 1)))


# ── Live Row Display ──────────────────────────────────────────────────────────

def display_live_row(row: dict, columns: list[str], row_num: int) -> None:
    """Display a single row as it's extracted (streaming mode)."""
    preview = []
    for col in columns[:4]:
        val = str(row.get(col, "N/A"))[:30]
        if val.lower() in ("n/a", "none", ""):
            preview.append(f"[dim]{col}: [red]N/A[/red][/dim]")
        else:
            preview.append(f"[dim]{col}: [green]{val}[/green][/dim]")
    console.print(f"  [cyan]Row {row_num}:[/cyan] {' │ '.join(preview)}")


# ── Final Summary Panel ──────────────────────────────────────────────────────

def display_final_summary(
    accepted: int,
    rejected: int,
    blocked: int,
    sources: int,
    iterations: int,
    null_rate: float,
    elapsed: float,
    output_path: str,
) -> None:
    """Display a beautiful final summary panel."""
    m, s = divmod(int(elapsed), 60)

    lines = [
        "",
        f"  [green]✓ Accepted rows:[/green]    [bold]{accepted:,}[/bold]",
        f"  [yellow]✗ Rejected rows:[/yellow]    [bold]{rejected:,}[/bold]",
        f"  [red]🛡 Blocked sources:[/red]  [bold]{blocked:,}[/bold]",
        f"  [cyan]📡 Sources scraped:[/cyan]  [bold]{sources:,}[/bold]",
        f"  [magenta]🔄 Iterations:[/magenta]      [bold]{iterations}[/bold]",
        f"  [blue]📉 Null rate:[/blue]        [bold]{null_rate:.0%}[/bold]",
        f"  [dim]⏱  Time elapsed:[/dim]    [bold]{m}:{s:02d}[/bold]",
        "",
        f"  [bold green]💾 Output:[/bold green] {output_path}",
        "",
    ]

    panel = Panel(
        "\n".join(lines),
        title="[bold bright_white on dark_green] ✨ Pipeline Complete ✨ [/bold bright_white on dark_green]",
        border_style="bright_green",
        padding=(0, 2),
        expand=False,
    )
    console.print()
    console.print(panel)
