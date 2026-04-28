"""
tools/image_chem_extractor.py — Chemical structure extraction from PDF images.

Extracts SMILES, SELFIES, InChI, and compound names from figures embedded
in PDFs using a cascade of strategies:

  Pass 1 : Caption-based LLM extraction  (text captions referencing structures)
  Pass 2 : DECIMER / RDKit OCR pipeline  (open-source structure recognition)
  Pass 3 : Claude Vision fallback        (if ANTHROPIC_API_KEY present)

Designed to slot into Agent 4 (ExtractionAgent) as an additional pass
for vision_columns like ["SMILES", "SELFIES", "Structure_InChI"].

Usage
-----
    from tools.image_chem_extractor import extract_chemical_structures_from_figures

    rows = extract_chemical_structures_from_figures(
        figures=doc["figures"],         # list of {path, page, caption}
        source_url=doc["source"],
        vision_columns=["SMILES", "SELFIES", "Compound_Name"],
        topic="antibody drug conjugates",
    )
    # rows → list of dicts, each keyed by vision_columns + "Source_URL"

File Locations
--------------
  Place this file at:   tools/image_chem_extractor.py
  Images are read from: data/images/<stem>/<figure_N>.png
  Results cached at:    data/chem_cache/<source_hash>_structures.json

Dependencies (install as needed)
---------------------------------
  pip install rdkit-pypi pillow                   # core
  pip install decimer                             # neural SMILES recognition
  pip install selfies                             # SMILES → SELFIES conversion
  pip install anthropic                           # vision fallback (already in stack)
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from tools.console_setup import console
import config as cfg

# ── Optional heavy imports — graceful fallback ────────────────────────────────

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from decimer import predict_SMILES  # type: ignore[import-untyped]
    DECIMER_AVAILABLE = True
except ImportError:
    DECIMER_AVAILABLE = False

try:
    import selfies as sf  # type: ignore[import-untyped]
    SELFIES_AVAILABLE = True
except ImportError:
    SELFIES_AVAILABLE = False

try:
    from rdkit import Chem  # type: ignore[import-untyped]
    from rdkit.Chem import AllChem, Draw  # type: ignore[import-untyped]
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    import anthropic as _anthropic  # type: ignore[import-untyped]
    VISION_AVAILABLE = bool(cfg.ANTHROPIC_API_KEY)
except ImportError:
    VISION_AVAILABLE = False

# ── Cache directory ───────────────────────────────────────────────────────────
CACHE_DIR = Path("data/chem_cache")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _smiles_to_selfies(smiles: str) -> str:
    """Convert a SMILES string to SELFIES. Returns '' on failure."""
    if not SELFIES_AVAILABLE or not smiles:
        return ""
    try:
        return sf.encoder(smiles)
    except Exception:
        return ""


def _validate_smiles(smiles: str) -> bool:
    """Return True if the SMILES string is chemically valid via RDKit."""
    if not RDKIT_AVAILABLE or not smiles:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def _smiles_to_inchi(smiles: str) -> str:
    """Convert SMILES to InChI string via RDKit."""
    if not RDKIT_AVAILABLE or not smiles:
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToInchi(mol) or ""
    except Exception:
        return ""
    return ""


def _load_cache(source_hash: str) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{source_hash}_structures.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache(source_hash: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{source_hash}_structures.json"
    cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _source_hash(source_url: str) -> str:
    return hashlib.md5(source_url.encode()).hexdigest()[:12]


# ── Pass 1: Caption-based extraction ─────────────────────────────────────────

_SMILES_PATTERN = re.compile(
    r'\b([A-Z][a-z]?(?:\d*[\(\)\[\]=#@+\-\\\/\.]*[A-Za-z0-9\(\)\[\]=#@+\-\\\/\.]*){4,})\b'
)

_INCHI_PATTERN = re.compile(r'InChI=\S+')

_COMPOUND_CONTEXT = re.compile(
    r'(?:compound|drug|conjugate|payload|linker|antibody|ADC|molecule|structure|'
    r'SMILES|formula)\s*[:\-–]?\s*([^\n\.]{3,80})',
    re.I
)


def _extract_from_caption(caption: str, vision_columns: list[str]) -> dict:
    """
    Heuristic extraction of chemical identifiers from figure captions.
    Returns a partial dict for the vision columns.
    """
    result: dict[str, str] = {}

    # SMILES-like token (rough heuristic — long alphanumeric with ring/bond chars)
    for m in _SMILES_PATTERN.finditer(caption):
        candidate = m.group(1)
        if _validate_smiles(candidate):
            if any(c in col.upper() for col in vision_columns for c in ["SMILES"]):
                result.setdefault("SMILES", candidate)
            if any("SELFIES" in col.upper() for col in vision_columns):
                result.setdefault("SELFIES", _smiles_to_selfies(candidate))
            if any("INCHI" in col.upper() for col in vision_columns):
                result.setdefault("Structure_InChI", _smiles_to_inchi(candidate))
            break

    # InChI string
    for m in _INCHI_PATTERN.finditer(caption):
        if any("INCHI" in col.upper() for col in vision_columns):
            result.setdefault("Structure_InChI", m.group(0))

    # Compound name context
    for m in _COMPOUND_CONTEXT.finditer(caption):
        name = m.group(1).strip().strip(".,;:")
        if 3 < len(name) < 80:
            for col in vision_columns:
                if any(kw in col.lower() for kw in ["name", "compound", "drug", "payload"]):
                    result.setdefault(col, name)
            break

    return result


# ── Pass 2: DECIMER neural structure recognition ──────────────────────────────

def _extract_with_decimer(image_path: str, vision_columns: list[str]) -> dict:
    """
    Run DECIMER on an image file to extract SMILES.
    Falls back gracefully if DECIMER is not installed.
    """
    if not DECIMER_AVAILABLE or not PIL_AVAILABLE:
        return {}
    if not os.path.exists(image_path):
        return {}
    try:
        smiles = predict_SMILES(image_path)
        if not smiles or not _validate_smiles(smiles):
            return {}
        result: dict[str, str] = {}
        for col in vision_columns:
            cu = col.upper()
            if "SMILES" in cu:
                result[col] = smiles
            elif "SELFIES" in cu:
                result[col] = _smiles_to_selfies(smiles)
            elif "INCHI" in cu:
                result[col] = _smiles_to_inchi(smiles)
        return result
    except Exception as e:
        console.print(f"  [dim yellow]DECIMER failed on {Path(image_path).name}: {e}[/dim yellow]")
        return {}


# ── Pass 3: Claude Vision fallback ───────────────────────────────────────────

_VISION_SYSTEM = (
    "You are a chemistry expert specialising in drug discovery and medicinal chemistry. "
    "Given a figure from a scientific paper, extract any chemical structures you can identify. "
    "Return ONLY a JSON object with these keys (use null for anything you cannot determine):\n"
    '  "SMILES": canonical SMILES string of the primary structure shown\n'
    '  "SELFIES": SELFIES encoding of that structure\n'
    '  "Structure_InChI": InChI string\n'
    '  "Compound_Name": compound name or identifier visible in the figure\n'
    '  "DAR": drug-antibody ratio if stated (numeric string)\n'
    '  "Payload": payload/warhead name if identifiable\n'
    '  "Linker_Type": linker type if identifiable\n'
    "Return ONLY the JSON object, no commentary."
)


def _extract_with_vision(image_path: str, caption: str, vision_columns: list[str]) -> dict:
    """
    Use Claude's vision API to extract chemical structure data from an image.
    """
    if not VISION_AVAILABLE:
        return {}
    if not os.path.exists(image_path):
        return {}
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        b64 = base64.standard_b64encode(img_bytes).decode()

        # Detect media type
        suffix = Path(image_path).suffix.lower()
        media_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(suffix, "image/png")

        client = _anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
        prompt = (
            f"Figure caption: {caption}\n\n"
            "Extract all chemical structure information visible in this figure."
        )
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=512,
            system=_VISION_SYSTEM,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        raw = response.content[0].text.strip()
        clean = re.sub(r'```(?:json)?\s*', '', raw).strip().strip('`')
        parsed = json.loads(clean)

        # Filter to requested vision_columns only
        result: dict[str, str] = {}
        for col in vision_columns:
            # Try exact key match first, then case-insensitive
            if col in parsed and parsed[col] not in (None, "", "null"):
                result[col] = str(parsed[col])
            else:
                for k, v in parsed.items():
                    if k.lower() == col.lower() and v not in (None, "", "null"):
                        result[col] = str(v)
                        break
        return result

    except Exception as e:
        console.print(f"  [dim yellow]Vision extraction failed: {e}[/dim yellow]")
        return {}


# ── Public Entry Point ────────────────────────────────────────────────────────

def extract_chemical_structures_from_figures(
    figures: list[dict],
    source_url: str,
    vision_columns: list[str],
    topic: str = "",
) -> list[dict]:
    """
    Run the 3-pass chemical structure extraction pipeline over a document's figures.

    Parameters
    ----------
    figures : list[dict]
        Each dict: {"path": str, "page": int, "caption": str}
        (produced by pdf_tools.extract_images)
    source_url : str
        Source URL or file path — used for cache keying and row tagging.
    vision_columns : list[str]
        Column names that require image-based extraction
        (e.g. ["SMILES", "SELFIES", "Compound_Name", "DAR"]).
    topic : str
        Research topic — used for context in vision prompts.

    Returns
    -------
    list[dict]
        Each dict maps vision_column → extracted value, plus "Source_URL".
        One dict per figure that yielded at least one non-empty value.
    """
    if not figures or not vision_columns:
        return []

    src_hash = _source_hash(source_url)
    cache    = _load_cache(src_hash)

    rows: list[dict] = []

    for fig in figures:
        image_path = fig.get("path", "")
        caption    = fig.get("caption", "")
        page       = fig.get("page", "?")
        fig_key    = f"page_{page}_{Path(image_path).name}"

        if fig_key in cache:
            cached_row = cache[fig_key]
            if cached_row:
                rows.append(cached_row)
            continue

        console.print(f"  [dim]🔬 Vision pass — {Path(image_path).name} (p.{page})[/dim]")

        merged: dict[str, str] = {}

        # Pass 1: caption heuristics (always run — fast)
        caption_result = _extract_from_caption(caption, vision_columns)
        merged.update(caption_result)

        # Pass 2: DECIMER (if image exists and not all columns filled)
        missing = [c for c in vision_columns if c not in merged]
        if missing and image_path:
            decimer_result = _extract_with_decimer(image_path, missing)
            merged.update(decimer_result)

        # Pass 3: Claude Vision (fallback for remaining missing columns)
        missing = [c for c in vision_columns if c not in merged]
        if missing and image_path and VISION_AVAILABLE:
            vision_result = _extract_with_vision(image_path, caption, missing)
            merged.update(vision_result)

        if merged:
            merged["Source_URL"] = source_url
            merged["Figure_Page"] = str(page)
            merged["Figure_Caption"] = caption[:200] if caption else ""
            rows.append(merged)
            cache[fig_key] = merged
            console.print(
                f"  [green]✓ Extracted {list(merged.keys())} from figure p.{page}[/green]"
            )
        else:
            cache[fig_key] = {}  # negative cache

    _save_cache(src_hash, cache)
    return rows


# ── Schema hint: which columns trigger vision extraction ─────────────────────

CHEMICAL_VISION_KEYWORDS = {
    "smiles", "selfies", "inchi", "structure", "mol", "compound",
    "dar", "drug_antibody_ratio", "payload_structure", "linker_structure",
    "chemical_formula", "molecular_formula",
}


def classify_vision_columns(columns: list[str]) -> list[str]:
    """
    Given a list of user-requested column names, return those that
    likely require image-based chemical structure extraction.

    Called by ExtractionAgent.__init__ or QueryArchitect to populate
    schema["requires_vision"].
    """
    return [
        col for col in columns
        if any(kw in col.lower().replace(" ", "_") for kw in CHEMICAL_VISION_KEYWORDS)
    ]
