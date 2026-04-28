"""
tools/llm_client.py — Unified LLM client: Claude API, OpenAI API, or Ollama.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import requests
from tools.console_setup import console

import config as cfg



# Phrases indicating an empty / not-found value
EMPTY_VALS = {
    "n/a", "none", "not specified", "not mentioned", "not provided",
    "not explicitly mentioned", "unspecified", "unknown", "na", "nil", "",
    "not available", "not found", "not discussed", "not applicable",
    "not stated", "not reported", "not described",
}


def is_empty(v: str) -> bool:
    return str(v).strip().lower() in EMPTY_VALS


def parse_json_response(text: str) -> list[dict] | None:
    """Robustly parse a JSON array or object from LLM response."""
    text = text.strip()

    # Strip preamble prose before first [ or {
    fb, bb = text.find('{'), text.find('[')
    starts = [i for i in [fb, bb] if i != -1]
    if starts:
        text = text[min(starts):]

    # Fix unquoted N/A
    text = re.sub(r':\s*N/A\s*([,}\]])', r': "N/A"\1', text)

    # Remove markdown fences
    text = re.sub(r'```(?:json)?\s*', '', text).strip().strip('`')

    for attempt in [text, text + ']', text + '}']:
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, list):
                return [d for d in parsed if isinstance(d, dict)]
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass

    # Last-ditch: find outermost array or object
    for bracket, closing in [('[', ']'), ('{', '}')]:
        si = text.find(bracket)
        if si == -1:
            continue
        ei = text.rfind(closing)
        if ei > si:
            try:
                parsed = json.loads(text[si:ei+1])
                if isinstance(parsed, list):
                    return [d for d in parsed if isinstance(d, dict)]
                if isinstance(parsed, dict):
                    return [parsed]
            except json.JSONDecodeError:
                pass
    return None


class LLMClient:
    """
    Unified LLM client. Provider is resolved from config.LLM_PROVIDER.
    """

    def __init__(self):
        self.provider = cfg.LLM_PROVIDER
        self.model    = cfg.active_model()
        self._client  = None
        self._cache: dict[str, Any] = {}  # prompt hash → response cache
        self._init()

    def _init(self):
        if self.provider == "claude":
            if not cfg.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
            import anthropic
            self._client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
            console.print(f"[green]✓ Claude API ready — {self.model}[/green]")

        elif self.provider == "openai":
            if not cfg.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
            import openai
            self._client = openai.OpenAI(api_key=cfg.OPENAI_API_KEY)
            console.print(f"[green]✓ OpenAI API ready — {self.model}[/green]")

        else:  # ollama
            try:
                import ollama as _ollama
                models = _ollama.list()
                available = [m.model for m in models.models]
                base = self.model.split(":")[0]
                if not any(base in m for m in available):
                    console.print(
                        f"[red]✗ Ollama model '{self.model}' not found.[/red]\n"
                        f"  Available: {available}\n"
                        f"  Run: ollama pull {self.model}"
                    )
                    raise SystemExit(1)
                console.print(f"[green]✓ Ollama ready — {self.model}[/green]")
            except ImportError:
                raise ImportError("Install ollama: pip install ollama")

    def complete(self, prompt: str, system: str = "", max_tokens: int = 4096) -> str:
        """Low-level completion — returns raw text."""
        try:
            if self.provider == "claude":
                msgs = [{"role": "user", "content": prompt}]
                r = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=msgs,
                )
                return r.content[0].text.strip()

            elif self.provider == "openai":
                msgs = []
                if system:
                    msgs.append({"role": "system", "content": system})
                msgs.append({"role": "user", "content": prompt})
                r = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=cfg.TEMPERATURE,
                    messages=msgs,
                )
                return r.choices[0].message.content.strip()

            else:  # ollama
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "system": system,
                    "stream": False,
                    "options": {
                        "num_gpu": cfg.NUM_GPU_LAYERS,
                        "num_ctx": cfg.CONTEXT_WINDOW,
                        "temperature": cfg.TEMPERATURE,
                    },
                }
                resp = requests.post(
                    f"{cfg.OLLAMA_URL}/api/generate",
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                return resp.json().get("response", "").strip()

        except Exception as e:
            console.print(f"  [yellow]⚠ LLM call failed: {e}[/yellow]")
            return ""

    def complete_json(self, prompt: str, system: str = "", max_tokens: int = 4096) -> Any:
        """Complete and parse as JSON. Returns parsed data or None."""
        raw = self.complete(prompt, system=system, max_tokens=max_tokens)
        if not raw:
            return None
        # Try cleaning markdown fences
        clean = re.sub(r'```(?:json)?\s*', '', raw).strip().strip('`')
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            return parse_json_response(raw)

    def get_embeddings(self, texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]] | None:
        """Get embeddings from OpenAI API. Returns None on failure."""
        try:
            import openai
            client = openai.OpenAI()
            truncated = [t[:8000] for t in texts]
            response = client.embeddings.create(input=truncated, model=model)
            return [item.embedding for item in response.data]
        except Exception as e:
            console.print(f"    [dim yellow]Embeddings unavailable ({e}), using structural scoring[/dim yellow]")
            return None
