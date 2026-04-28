"""
config.py — Central configuration for the Multi-Agent Dataset Builder.
All tunable constants live here. Loaded from .env automatically.
"""

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# ── LLM Provider ──────────────────────────────────────────────────────────────
LLM_PROVIDER      = os.getenv("LLM_PROVIDER",   "openai").lower()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = os.getenv("CLAUDE_MODEL",      "claude-sonnet-4-20250514")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL      = os.getenv("OPENAI_MODEL",   "gpt-4o-mini")
OLLAMA_URL        = os.getenv("OLLAMA_URL",      "http://localhost:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL",    "llama3.1:8b-instruct-q4_0")
NUM_GPU_LAYERS    = int(os.getenv("NUM_GPU_LAYERS", "99"))
CONTEXT_WINDOW    = int(os.getenv("CONTEXT_WINDOW", "8192"))

# ── Extraction ────────────────────────────────────────────────────────────────
MAX_TEXT_CHARS   = int(os.getenv("MAX_TEXT_CHARS",  "12000"))
MAX_RETRIES      = int(os.getenv("MAX_RETRIES",     "2"))
TEMPERATURE      = float(os.getenv("TEMPERATURE",   "0.0"))
SCHEMA_TEMPERATURE = float(os.getenv("SCHEMA_TEMPERATURE", "0.2"))

# ── Scraping ──────────────────────────────────────────────────────────────────
DEFAULT_LIMIT    = int(os.getenv("SCRAPE_LIMIT",   "10"))
DEFAULT_WORKERS  = int(os.getenv("SCRAPE_WORKERS", "2"))
REQUEST_DELAY    = (
    int(os.getenv("REQUEST_DELAY_MIN", "2")),
    int(os.getenv("REQUEST_DELAY_MAX", "5")),
)

# ── Anti-bot ──────────────────────────────────────────────────────────────────
CURL_IMPERSONATE     = os.getenv("CURL_IMPERSONATE", "chrome")
MAX_BACKOFF_RETRIES  = int(os.getenv("MAX_BACKOFF_RETRIES", "5"))
BACKOFF_BASE_SECONDS = float(os.getenv("BACKOFF_BASE_SECONDS", "1.5"))
# Proxy list (comma-separated: http://user:pass@host:port)
PROXY_LIST_RAW = os.getenv("PROXY_LIST", "")
PROXY_LIST: list[str] = [p.strip() for p in PROXY_LIST_RAW.split(",") if p.strip()]

# ── Source counting: max_sources = PDF count only ─────────────────────────────
# The user's requested source count refers to PDFs (full papers).
# HTML sources are collected in addition, not counted against the limit.
PDF_SOURCES_LIMIT = int(os.getenv("PDF_SOURCES_LIMIT", str(DEFAULT_LIMIT)))

# ── Workspace ─────────────────────────────────────────────────────────────────
PROJECT_ROOT_DIR = os.getenv("PROJECT_ROOT_DIR", "data")
OUTPUT_DIR       = os.getenv("OUTPUT_DIR",  "data/outputs")
PDF_TEMP_DIR     = os.getenv("PDF_DIR",     "data/pdfs")
CHUNK_TEMP_DIR   = os.getenv("CHUNK_DIR",   "data/chunks")
SUPPLEMENT_DIR   = "data/supplements"
MEMORY_DIR       = os.getenv("MEMORY_DIR",  "data/memory")
IMAGE_DIR        = os.getenv("IMAGE_DIR",   "data/images")
TABLE_DIR        = os.getenv("TABLE_DIR",   "data/tables")
OUTPUT_FILE      = os.path.join(OUTPUT_DIR, "output.csv")

# ── OCR ───────────────────────────────────────────────────────────────────────
ENABLE_OCR    = os.getenv("ENABLE_OCR", "true").lower() in ("true", "1", "yes")
TESSERACT_CMD = os.getenv("TESSERACT_CMD") or shutil.which("tesseract") or "tesseract"

# ── Vision ────────────────────────────────────────────────────────────────────
ENABLE_VISION = os.getenv("ENABLE_VISION", "false").lower() in ("true", "1", "yes")
MIN_IMAGE_AREA_PX = int(os.getenv("MIN_IMAGE_AREA_PX", "10000"))  # 100×100 px

# ── Agent tuning ──────────────────────────────────────────────────────────────
MAX_ADAPTIVE_LOOPS = int(os.getenv("MAX_ADAPTIVE_LOOPS", "3"))
MAX_NULL_RATE      = float(os.getenv("MAX_NULL_RATE",    "0.35"))
MIN_YIELD_RATE     = float(os.getenv("MIN_YIELD_RATE",   "0.35"))
OLLAMA_MAX_CHUNKS  = int(os.getenv("OLLAMA_MAX_CHUNKS",  "6"))
NUM_QUERIES        = int(os.getenv("NUM_QUERIES",        "15"))

# ── Science APIs (for NullHunter) ─────────────────────────────────────────────
PUBCHEM_API_URL   = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CROSSREF_API_URL  = "https://api.crossref.org/works"
PUBMED_API_URL    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CHEMBL_API_URL    = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT_API_URL   = "https://rest.uniprot.org/uniprotkb"
OPENTARGETS_URL   = "https://api.platform.opentargets.org/api/v4/graphql"
# NCBI email (required for Entrez API polite access)
NCBI_EMAIL        = os.getenv("NCBI_EMAIL", "pipeline@research.local")

# ── Auth ──────────────────────────────────────────────────────────────────────
ENABLE_AUTH = os.getenv("ENABLE_AUTH", "false").lower() in ("true", "1", "yes")

# ── Debug ─────────────────────────────────────────────────────────────────────
DEBUG = os.getenv("DEBUG", "").lower() in ("true", "1", "yes")


def active_model() -> str:
    if LLM_PROVIDER == "claude":  return CLAUDE_MODEL
    if LLM_PROVIDER == "openai":  return OPENAI_MODEL
    return OLLAMA_MODEL


# ── Topic workspace helpers ───────────────────────────────────────────────────

def _sanitize_topic(topic: str) -> str:
    name = topic.strip().replace(" ", "_")
    name = "".join(c for c in name if c.isalnum() or c in ("_", "-"))
    return name[:60] or "default"


def get_topic_paths(topic: str, root: str | None = None) -> dict[str, str]:
    root = root or PROJECT_ROOT_DIR
    slug = _sanitize_topic(topic)
    base = os.path.join(root, slug)
    out_dir = os.path.join(base, "outputs")
    return {
        "base_dir":        base,
        "output_dir":      out_dir,
        "sources_dir":     os.path.join(base, "pdfs"),
        "chunks_dir":      os.path.join(base, "chunks"),
        "images_dir":      os.path.join(base, "images"),
        "tables_dir":      os.path.join(base, "tables"),
        "memory_dir":      os.path.join(base, "memory"),
        "supplements_dir": os.path.join(base, "supplements"),
        "output_file":     os.path.join(out_dir, "output.csv"),
        "checkpoint_file": os.path.join(out_dir, "checkpoint.json"),
    }


def init_topic_workspace(topic: str, root: str | None = None) -> dict[str, str]:
    paths = get_topic_paths(topic, root)
    for key in ("output_dir", "sources_dir", "chunks_dir",
                "images_dir", "tables_dir", "memory_dir", "supplements_dir"):
        Path(paths[key]).mkdir(parents=True, exist_ok=True)
    return paths


def apply_topic_paths(topic: str, root: str | None = None) -> dict[str, str]:
    global OUTPUT_DIR, PDF_TEMP_DIR, CHUNK_TEMP_DIR, SUPPLEMENT_DIR, \
           MEMORY_DIR, IMAGE_DIR, TABLE_DIR, OUTPUT_FILE
    paths = init_topic_workspace(topic, root)
    OUTPUT_DIR     = paths["output_dir"]
    PDF_TEMP_DIR   = paths["sources_dir"]
    CHUNK_TEMP_DIR = paths["chunks_dir"]
    SUPPLEMENT_DIR = paths["supplements_dir"]
    MEMORY_DIR     = paths["memory_dir"]
    IMAGE_DIR      = paths["images_dir"]
    TABLE_DIR      = paths["tables_dir"]
    OUTPUT_FILE    = paths["output_file"]
    return paths
