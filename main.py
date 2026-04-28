"""
main.py — Multi-Agent Dataset Builder (unified entry point)
============================================================

TWO interaction modes:

1. CHATBOT (default — recommended):
   python main.py
   python main.py --chat

2. BATCH MODE (traditional CLI):
   python main.py --batch --config sample_config.json
   python main.py --batch --topic "antibody drug conjugates"
   python main.py --batch --topic "electric vehicles" --fields "Model,Brand,Range,Price"
   python main.py --batch --topic "CRISPR" --pdfs paper1.pdf paper2.pdf
   python main.py --batch --topic "AI safety" --pdf-folder ./papers/
"""

import argparse
import json
import os
import sys
from pathlib import Path

from tools.console_setup import console
from rich.panel import Panel
from rich.prompt import Prompt

import config as cfg
from tools.llm_client import LLMClient
from agents.orchestrator import run_pipeline_from_config




def load_config(config_path: str) -> dict:
    """Load and validate a JSON/YAML config file."""
    path = Path(config_path)
    if not path.exists():
        console.print(f"[red]✗ Config file not found: {config_path}[/red]")
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix in (".yaml", ".yml"):
                import yaml
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
    except Exception as e:
        console.print(f"[red]✗ Failed to parse config: {e}[/red]")
        sys.exit(1)

    if not config.get("dataset_name") and not config.get("description"):
        console.print("[red]✗ Config must have 'dataset_name' or 'description'[/red]")
        sys.exit(1)

    console.print(f"[green]✓ Config loaded: {config.get('dataset_name', 'unnamed')}[/green]")
    return config


def build_config_from_cli(args) -> dict:
    """Convert CLI arguments into a config dict."""
    config = {
        "dataset_name": args.topic,
        "description": args.topic,
        "min_rows": 50,
        "max_sources": args.limit,
        "quality_threshold": "high",
        "max_adaptive_loops": 3,
        "output_path": args.output,
    }

    if args.fields:
        config["columns"] = [{"name": f.strip()} for f in args.fields.split(",") if f.strip()]

    return config


def collect_pdfs(pdf_args: list[str] | None, folder: str | None) -> list[str]:
    """Gather PDF paths from --pdfs and/or --pdf-folder."""
    paths: list[str] = []
    if pdf_args:
        for p in pdf_args:
            if os.path.isfile(p):
                paths.append(p)
            else:
                console.print(f"[yellow]⚠ File not found: {p}[/yellow]")
    if folder:
        folder_path = Path(folder)
        if not folder_path.is_dir():
            console.print(f"[red]✗ Folder not found: {folder}[/red]")
        else:
            found = sorted(folder_path.rglob("*.pdf"))
            console.print(f"[green]Found {len(found)} PDF(s) in {folder}[/green]")
            paths.extend(str(p) for p in found)
    return paths


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-Agent Dataset Builder — extract structured data from any topic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode selection
    p.add_argument("--chat",       action="store_true", default=True,
                   help="Launch interactive chatbot (default)")
    p.add_argument("--batch",      action="store_true", default=False,
                   help="Run in batch mode (traditional CLI)")

    # Input modes (batch only)
    p.add_argument("--config",     "-c", default=None, help="Path to JSON/YAML config file")
    p.add_argument("--topic",      "-t", default=None, help="Research topic (batch mode)")

    # Batch mode options
    p.add_argument("--fields",     "-f", default=None,  help="Comma-separated field names")
    p.add_argument("--limit",      "-n", type=int, default=cfg.DEFAULT_LIMIT,
                   help="Max sources to process (default: %(default)s)")
    p.add_argument("--urls",       "-u", nargs="*",     help="Specific URLs to scrape")
    p.add_argument("--pdfs",       "-p", nargs="*",     help="Local PDF file paths")
    p.add_argument("--pdf-folder", "-d", default=None,  help="Folder containing PDFs")
    p.add_argument("--output",     "-o", default=cfg.OUTPUT_FILE, help="Output CSV path")

    # General options
    p.add_argument("--provider",   choices=["claude", "openai", "ollama"], default=None,
                   help="LLM provider (overrides .env)")
    p.add_argument("--auto",       action="store_true", help="Skip interactive prompts (batch)")
    p.add_argument("--min-rows",   type=int, default=50, help="Minimum rows target")
    return p.parse_args()


def main():
    args = parse_args()

    # Override provider if requested
    if args.provider:
        cfg.LLM_PROVIDER = args.provider

    # ── Chatbot mode (default) ────────────────────────────────────────────────
    if not args.batch and not args.config and not args.topic:
        from chatbot import run_chatbot
        run_chatbot()
        return

    # ── Batch mode ────────────────────────────────────────────────────────────
    BANNER = """[bold cyan]╔══════════════════════════════════════════════╗[/bold cyan]
[bold cyan]║  Multi-Agent Dataset Builder  (Batch Mode)   ║[/bold cyan]
[bold cyan]║  Search · Scrape · Extract · Validate · CSV  ║[/bold cyan]
[bold cyan]╚══════════════════════════════════════════════╝[/bold cyan]
[dim]Providers: Claude | OpenAI | Ollama  |  curl_cffi stealth[/dim]"""

    console.print(BANNER)

    if not args.config and not args.topic:
        console.print("[red]✗ Batch mode requires --config <file> or --topic <text>[/red]")
        console.print("[dim]  Or just run without flags for chatbot mode![/dim]")
        sys.exit(1)

    if args.config:
        config = load_config(args.config)
    else:
        config = build_config_from_cli(args)

    console.print(f"\n[bold]Topic:[/bold] {config.get('description', config.get('dataset_name'))}")
    console.print(f"[bold]Provider:[/bold] {cfg.LLM_PROVIDER} / {cfg.active_model()}\n")

    # Initialize LLM client
    try:
        client = LLMClient()
    except (ValueError, SystemExit) as e:
        console.print(f"[red]✗ {e}[/red]")
        sys.exit(1)

    # Handle columns — in batch mode, use fields from CLI or config
    if not config.get("columns") and args.fields:
        config["columns"] = [{"name": f.strip()} for f in args.fields.split(",") if f.strip()]

    if args.min_rows:
        config["min_rows"] = args.min_rows

    # Collect local PDFs
    local_pdfs = collect_pdfs(args.pdfs, args.pdf_folder) if (args.pdfs or args.pdf_folder) else None
    if args.urls:
        local_pdfs = None

    # Override output
    if args.output:
        config["output_path"] = args.output

    # ── Handle direct URLs: inject into config so the pipeline seeds them ──────
    if args.urls and not local_pdfs:
        config["_inject_urls"] = args.urls

    # ── Run pipeline ─────────────────────────────────────────────────────────
    result = run_pipeline_from_config(config, local_pdfs=local_pdfs or args.urls)

    console.print(f"\n[bold green]Done![/bold green] Output saved to [cyan]{result.output_path}[/cyan]")


if __name__ == "__main__":
    main()
