"""
tools/pdf_tools.py — PDF extraction using Docling as the primary engine.

Extraction cascade:
  1. Docling  — primary engine. Handles digital, scanned, rotated, multi-column,
               complex academic layouts. Uses AI layout models internally.
               Catches horizontal (sideways) tables via its vision pipeline.
  2. pdfplumber — fallback for tables if Docling extracts 0 tables.
  3. PyMuPDF   — fallback text extraction and image extraction.
  4. pytesseract — last-resort OCR for fully scanned pages.

All existing public signatures are preserved:
    extract_text(pdf_path)          → str
    extract_text_ocr(pdf_path)      → str
    extract_tables(pdf_path)        → list[dict]
    extract_images(pdf_path, dir)   → list[dict]
    clean_text(text)                → str
    chunk_text(text, ...)           → list[str]
    is_duplicate_pdf(pdf_path)      → bool
    reset_seen_hashes()

New additions:
    extract_all(pdf_path, image_dir) → DoclingResult  (single-pass full extraction)

Rotated / horizontal table handling:
  Docling's EasyOCR + layout model detects tables regardless of orientation.
  When do_ocr=True (ENABLE_OCR=true in .env), it rasterises pages and runs
  a two-stage cell classifier that correctly handles 90°/270° rotated tables.
  For digital (non-scanned) PDFs, Docling's PDF backend uses geometric cell
  detection that is orientation-independent.

Install:
    pip install docling docling-core
    pip install pdfplumber pymupdf  # kept as fallbacks
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
console = Console()
import config as cfg


# ── Duplicate detection ───────────────────────────────────────────────────────

_seen_hashes: set[str] = set()


def is_duplicate_pdf(pdf_path: str) -> bool:
    """Return True if this PDF was already processed (by SHA-256 content hash)."""
    try:
        with open(pdf_path, "rb") as fh:
            h = hashlib.sha256(fh.read()).hexdigest()
        if h in _seen_hashes:
            console.print("  [dim]⊘ Duplicate PDF — skipping[/dim]")
            return True
        _seen_hashes.add(h)
        return False
    except Exception:
        return False


def reset_seen_hashes() -> None:
    _seen_hashes.clear()


# ── Docling import (graceful) ─────────────────────────────────────────────────

try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
        EasyOcrOptions,
        TesseractCliOcrOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc.labels import DocItemLabel
    _DOCLING_AVAILABLE = True
except ImportError:
    _DOCLING_AVAILABLE = False
    console.print(
        "  [yellow]⚠ docling not installed — using pdfplumber/PyMuPDF fallbacks.\n"
        "    Install: pip install docling docling-core[/yellow]"
    )


# ── Result dataclass returned by extract_all() ───────────────────────────────

@dataclass
class DoclingResult:
    """Unified extraction result from a single Docling pass."""
    full_text:   str             = ""
    text_blocks: list[str]       = field(default_factory=list)
    tables:      list[dict]      = field(default_factory=list)
    figures:     list[dict]      = field(default_factory=list)
    page_count:  int             = 0
    method:      str             = "unknown"
    warnings:    list[str]       = field(default_factory=list)


# ── Junk section stripper ─────────────────────────────────────────────────────

_JUNK_SECTION_RE = re.compile(
    r"\n(?:References|Bibliography|Acknowledgements?|Acknowledgments?|"
    r"Funding|Conflict[s]? of Interest|Author Contributions?|"
    r"Supporting Information|Abbreviations|Ethical Approval|"
    r"Data Availability Statement|Supplementary (?:Data|Material|Information))"
    r"[\s\n]{0,60}",
    re.IGNORECASE,
)

_JUNK_HEADERS_SET = {
    "references", "bibliography", "acknowledgment", "acknowledgement",
    "funding", "conflict of interest", "author contribution",
    "supplementary", "abbreviation", "ethical approval",
    "data availability", "supporting information",
}


def _strip_junk_sections(text: str) -> str:
    """Truncate text at the first junk section header (References, Ack, etc.)."""
    match = _JUNK_SECTION_RE.search(text)
    return text[:match.start()].strip() if match else text


# ── Docling pipeline builder ──────────────────────────────────────────────────

def _build_docling_converter(
    do_ocr: bool = False,
    force_full_page_ocr: bool = False,
) -> "DocumentConverter":
    """
    Build a Docling DocumentConverter configured for academic PDFs.

    Parameters
    ----------
    do_ocr : bool
        Enable OCR. Required for scanned pages and rotated tables.
        Uses EasyOCR by default; falls back to Tesseract CLI if EasyOCR fails.
    force_full_page_ocr : bool
        Force OCR on every page even if text layer is detected.
        Use this when a PDF has a text layer but rotated/embedded tables
        that the text layer misses.
    """
    opts = PdfPipelineOptions()

    # ── Table extraction ──────────────────────────────────────────────────────
    # ACCURATE mode uses TableFormer (deep learning) — handles complex,
    # merged-cell, borderless, and rotated tables.
    # FAST mode uses heuristics — faster but misses hard cases.
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.table_structure_options.do_cell_matching = True   # align cells to text spans

    # ── OCR ───────────────────────────────────────────────────────────────────
    # In Docling v2, force_full_page_ocr is set within the OCR engine options,
    # not directly on the PipelineOptions object.
    opts.do_ocr = do_ocr

    if do_ocr:
        # EasyOCR: handles rotated text natively (angle detection built in).
        # Fallback to Tesseract CLI if EasyOCR is not installed.
        try:
            import easyocr  # noqa: F401
            opts.ocr_options = EasyOcrOptions(
                force_full_page_ocr=force_full_page_ocr,
                lang=["en"],
            )
            console.print("  [dim]OCR engine: EasyOCR (rotation-aware)[/dim]")
        except ImportError:
            opts.ocr_options = TesseractCliOcrOptions(
                force_full_page_ocr=force_full_page_ocr,
                lang="eng",
                tesseract_cmd=cfg.TESSERACT_CMD,
            )
            console.print("  [dim]OCR engine: Tesseract CLI[/dim]")

    # ── Image extraction ──────────────────────────────────────────────────────
    opts.generate_picture_images = True   # PIL Image objects embedded in result
    opts.images_scale = 2.0              # 2× resolution for sharper figure images

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=opts)
        }
    )


# ── Main single-pass extraction ───────────────────────────────────────────────

def extract_all(
    pdf_path: str,
    image_dir: str = "",
    do_ocr: bool | None = None,
    force_full_page_ocr: bool = False,
) -> DoclingResult:
    """
    Single-pass full extraction using Docling.

    Returns a DoclingResult with:
      - full_text    : cleaned body text (junk sections stripped)
      - text_blocks  : list of paragraph/section strings for chunking
      - tables       : list of table dicts (csv, header, page, caption, rows, cols)
      - figures      : list of figure dicts (path, page, caption, width, height)
      - page_count   : int
      - method       : "docling" | "docling+ocr" | "fallback"
      - warnings     : list of warning strings

    Rotated/horizontal table strategy
    ----------------------------------
    Digital PDFs (text layer present):
      Docling's geometric PDF backend detects cell boundaries from PDF vector
      graphics regardless of page rotation. A table rotated 90° is extracted
      correctly because cell coordinates are layout-based, not text-order-based.

    Scanned / image-only PDFs:
      When do_ocr=True, Docling rasterises pages at 2× scale, runs EasyOCR
      which has built-in angle detection, and feeds detected text blocks to
      TableFormer for structure recovery. This handles 90° and 270° rotated
      tables in scanned papers.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    image_dir : str
        Directory to save extracted figure images. If empty, images are not saved.
    do_ocr : bool | None
        None = auto-detect (use cfg.ENABLE_OCR). True/False = explicit override.
    force_full_page_ocr : bool
        Force full-page OCR even on digital PDFs (useful for embedded rotated tables).
    """
    if not _DOCLING_AVAILABLE:
        return _fallback_extract_all(pdf_path, image_dir)

    if do_ocr is None:
        do_ocr = cfg.ENABLE_OCR

    result = DoclingResult()
    result.method = "docling+ocr" if do_ocr else "docling"

    try:
        converter = _build_docling_converter(
            do_ocr=do_ocr,
            force_full_page_ocr=force_full_page_ocr,
        )

        console.print(f"  [dim]Docling: converting {Path(pdf_path).name}…[/dim]")
        t0 = time.time()
        conv_result = converter.convert(pdf_path)
        doc = conv_result.document
        elapsed = time.time() - t0
        console.print(f"  [dim]Docling: done in {elapsed:.1f}s[/dim]")

        result.page_count = getattr(doc, "num_pages", 0) or _count_pdf_pages(pdf_path)

        # ── Collect text, tables, figures in reading order ────────────────
        raw_text_parts: list[str] = []
        table_index = 0

        for item, _ in doc.iterate_items():
            label = getattr(item, "label", None)

            # ── Text blocks ───────────────────────────────────────────────
            if label in (
                DocItemLabel.PARAGRAPH,
                DocItemLabel.TEXT,
                DocItemLabel.SECTION_HEADER,
                DocItemLabel.TITLE,
                DocItemLabel.CAPTION,
                DocItemLabel.LIST_ITEM,
            ):
                text = getattr(item, "text", "").strip()
                if not text:
                    continue
                # Skip junk-section headers
                if text.lower() in _JUNK_HEADERS_SET:
                    break   # stop collecting text after junk section starts
                raw_text_parts.append(text)

            # ── Tables ────────────────────────────────────────────────────
            elif label == DocItemLabel.TABLE:
                tbl_dict = _extract_docling_table(item, doc, table_index, pdf_path)
                if tbl_dict:
                    result.tables.append(tbl_dict)
                    table_index += 1
                    console.print(
                        f"  [dim]Table {table_index}: "
                        f"{tbl_dict['rows']}×{tbl_dict['cols']} "
                        f"p{tbl_dict['page']}[/dim]"
                    )

            # ── Figures ───────────────────────────────────────────────────
            elif label == DocItemLabel.PICTURE:
                fig_dict = _extract_docling_figure(item, doc, image_dir, len(result.figures))
                if fig_dict:
                    result.figures.append(fig_dict)

        # ── Assemble full text ────────────────────────────────────────────
        raw_text = "\n\n".join(raw_text_parts)
        result.full_text = _strip_junk_sections(clean_text(raw_text))
        result.text_blocks = chunk_text(result.full_text) if result.full_text else []

        # ── Fallback: if Docling got 0 tables, try pdfplumber ────────────
        if not result.tables:
            console.print(
                "  [dim yellow]Docling found 0 tables — trying pdfplumber fallback[/dim yellow]"
            )
            fallback_tables = _pdfplumber_tables(pdf_path)
            if fallback_tables:
                result.tables = fallback_tables
                result.warnings.append("tables_via_pdfplumber_fallback")
                console.print(f"  [dim]pdfplumber: {len(fallback_tables)} table(s)[/dim]")

        # ── Fallback: if Docling got no text, use PyMuPDF ────────────────
        if not result.full_text:
            console.print(
                "  [dim yellow]Docling got no text — trying PyMuPDF fallback[/dim yellow]"
            )
            result.full_text = _pymupdf_text(pdf_path)
            result.text_blocks = chunk_text(result.full_text) if result.full_text else []
            result.warnings.append("text_via_pymupdf_fallback")

        return result

    except Exception as e:
        console.print(f"  [yellow]⚠ Docling failed: {e} — using fallback[/yellow]")
        result = _fallback_extract_all(pdf_path, image_dir)
        result.warnings.append(f"docling_failed:{e}")
        return result


# ── Table extraction from a Docling TableItem ─────────────────────────────────

def _extract_docling_table(
    item: Any,
    doc: Any,
    index: int,
    pdf_path: str,
) -> dict | None:
    """
    Convert a Docling TableItem into the pipeline's table dict format.

    Returns
    -------
    dict with keys: csv, header, page, rows, cols, caption, source
    Returns None if the table has fewer than 2 rows or 2 columns.
    """
    try:
        # Export to pandas DataFrame
        df = item.export_to_dataframe(doc)
    except Exception:
        # Fallback: export to markdown then parse
        try:
            md = item.export_to_markdown(doc)
            df = _markdown_table_to_df(md)
        except Exception:
            return None

    if df is None or df.shape[0] < 1 or df.shape[1] < 2:
        return None

    # Clean: strip whitespace, replace None
    df = df.fillna("").astype(str)
    df = df.map(lambda x: x.strip())

    # Remove fully empty rows and columns
    df = df.loc[~(df == "").all(axis=1)]
    df = df.loc[:, ~(df == "").all(axis=0)]

    if df.shape[0] < 1 or df.shape[1] < 2:
        return None

    header = list(df.columns)

    # Build CSV string — quote all values to handle commas inside cells
    rows_data = [header] + df.values.tolist()
    csv_lines = [",".join(f'"{str(cell)}"' for cell in row) for row in rows_data]

    # Try to find caption
    caption = _find_docling_caption(item, doc, index)
    if not caption:
        caption = _find_table_caption(pdf_path, getattr(item, "page_no", 0) or 0)

    page_no = getattr(item, "page_no", None)
    if page_no is None:
        try:
            page_no = item.prov[0].page_no if item.prov else 0
        except Exception:
            page_no = 0

    return {
        "csv":     f"[Table {index+1} page {page_no}]" +
                   (f" {caption}" if caption else "") +
                   "\n" + "\n".join(csv_lines),
        "header":  header,
        "page":    page_no,
        "rows":    df.shape[0],
        "cols":    df.shape[1],
        "caption": caption,
        "source":  "docling",
        "df":      df,   # keep DataFrame for downstream numeric merge
    }


def _find_docling_caption(item: Any, doc: Any, index: int) -> str:
    """Extract caption text from a Docling table item if available."""
    try:
        # Docling sometimes stores captions as a text reference
        if hasattr(item, "caption_text") and callable(item.caption_text):
            return item.caption_text(doc).strip()
        if hasattr(item, "caption") and item.caption:
            cap = item.caption
            if isinstance(cap, str):
                return cap.strip()
            if hasattr(cap, "text"):
                return cap.text.strip()
    except Exception:
        pass
    return ""


def _markdown_table_to_df(md: str):
    """Parse a Markdown table string into a pandas DataFrame."""
    try:
        import pandas as pd
        lines = [l.strip() for l in md.strip().splitlines() if l.strip().startswith("|")]
        if len(lines) < 2:
            return None
        # Strip separator row (|---|---|)
        lines = [l for l in lines if not re.match(r"^\|[\s\-:|]+\|$", l)]
        rows = [[cell.strip() for cell in l.strip("|").split("|")] for l in lines]
        if not rows:
            return None
        return pd.DataFrame(rows[1:], columns=rows[0])
    except Exception:
        return None


# ── Figure extraction from a Docling PictureItem ─────────────────────────────

def _extract_docling_figure(
    item: Any,
    doc: Any,
    image_dir: str,
    index: int,
) -> dict | None:
    """
    Extract a figure image from a Docling PictureItem.
    Filters out images below cfg.MIN_IMAGE_AREA_PX.
    """
    try:
        pil_img = item.get_image(doc)
        if pil_img is None:
            return None

        w, h = pil_img.size
        if w * h < cfg.MIN_IMAGE_AREA_PX:
            console.print(f"  [dim]⊘ Tiny figure skipped ({w}×{h})[/dim]")
            return None

        caption = ""
        try:
            if hasattr(item, "caption_text") and callable(item.caption_text):
                caption = item.caption_text(doc).strip()
        except Exception:
            pass

        page_no = 0
        try:
            page_no = item.prov[0].page_no if item.prov else 0
        except Exception:
            pass

        fig_dict: dict = {
            "page":    page_no,
            "caption": caption,
            "width":   w,
            "height":  h,
            "path":    "",
        }

        if image_dir:
            os.makedirs(image_dir, exist_ok=True)
            fname = f"figure_{index+1}_p{page_no}.png"
            fpath = os.path.join(image_dir, fname)
            pil_img.save(fpath, format="PNG")
            fig_dict["path"] = fpath

        return fig_dict

    except Exception as e:
        console.print(f"  [dim yellow]Figure extraction error: {e}[/dim yellow]")
        return None


# ── Public API — backward-compatible ─────────────────────────────────────────

def extract_text(pdf_path: str) -> str:
    """
    Extract full body text from a PDF.

    Uses Docling as primary engine. Falls back to PyMuPDF layout-aware
    extraction, then bare get_text() if needed.
    Returns '' if duplicate.
    """
    if is_duplicate_pdf(pdf_path):
        return ""

    # Docling path
    if _DOCLING_AVAILABLE:
        try:
            result = extract_all(pdf_path, image_dir="", do_ocr=False)
            if result.full_text and len(result.full_text.strip()) >= 300:
                return result.full_text
            # Text was too short — retry with OCR
            if cfg.ENABLE_OCR:
                console.print("  [dim yellow]Short text — retrying with OCR…[/dim yellow]")
                result_ocr = extract_all(pdf_path, image_dir="", do_ocr=True)
                if result_ocr.full_text:
                    return result_ocr.full_text
        except Exception as e:
            console.print(f"  [dim yellow]Docling text failed: {e}[/dim yellow]")

    # PyMuPDF fallback
    text = _pymupdf_text(pdf_path)
    if text:
        return text

    console.print("  [yellow]⚠ All text extraction methods failed[/yellow]")
    return ""


def extract_text_ocr(pdf_path: str) -> str:
    """
    OCR-based text extraction for fully scanned / image-only PDFs.
    Tries Docling+EasyOCR first, then pytesseract as last resort.
    """
    if not cfg.ENABLE_OCR:
        return ""

    console.print("  [yellow]Starting OCR extraction…[/yellow]")

    # Docling + OCR (handles rotation)
    if _DOCLING_AVAILABLE:
        try:
            result = extract_all(pdf_path, image_dir="", do_ocr=True, force_full_page_ocr=True)
            if result.full_text and len(result.full_text.strip()) >= 100:
                return result.full_text
        except Exception as e:
            console.print(f"  [dim yellow]Docling OCR failed: {e}[/dim yellow]")

    # pytesseract last resort
    return _pytesseract_ocr(pdf_path)


def extract_tables(pdf_path: str) -> list[dict]:
    """
    Extract all tables from a PDF.

    Cascade:
      1. Docling (TableFormer ACCURATE — handles rotated, merged-cell, borderless)
      2. pdfplumber (fallback for simple bordered tables)
      3. If still empty: Docling with OCR enabled (for scanned pages)

    Returns list of dicts:
      {csv, header, page, rows, cols, caption, source}
    """
    if _DOCLING_AVAILABLE:
        try:
            result = extract_all(pdf_path, image_dir="", do_ocr=False)
            if result.tables:
                return result.tables

            # No tables found without OCR — try with OCR for scanned/rotated
            if cfg.ENABLE_OCR:
                console.print(
                    "  [dim yellow]No tables without OCR — retrying with OCR "
                    "(handles rotated tables)…[/dim yellow]"
                )
                result_ocr = extract_all(
                    pdf_path, image_dir="", do_ocr=True, force_full_page_ocr=False
                )
                if result_ocr.tables:
                    return result_ocr.tables
        except Exception as e:
            console.print(f"  [dim yellow]Docling table extraction failed: {e}[/dim yellow]")

    # pdfplumber fallback
    return _pdfplumber_tables(pdf_path)


def extract_images(pdf_path: str, output_dir: str) -> list[dict]:
    """
    Extract figures from PDF, filtering out small/irrelevant images.

    Uses Docling's picture extraction first (better figure/table boundary
    detection), then PyMuPDF as fallback.

    Returns list of dicts:
      {path, page, caption, width, height}
    """
    os.makedirs(output_dir, exist_ok=True)
    results: list[dict] = []

    if _DOCLING_AVAILABLE:
        try:
            result = extract_all(pdf_path, image_dir=output_dir, do_ocr=False)
            if result.figures:
                return result.figures
        except Exception as e:
            console.print(f"  [dim yellow]Docling image extraction failed: {e}[/dim yellow]")

    # PyMuPDF fallback
    return _pymupdf_images(pdf_path, output_dir)


# ── Internal fallbacks (pdfplumber, PyMuPDF, tesseract) ──────────────────────

def _pdfplumber_tables(pdf_path: str) -> list[dict]:
    """
    pdfplumber table extraction with lattice + stream strategies.
    Used as fallback when Docling finds 0 tables.
    """
    results: list[dict] = []
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for pnum, page in enumerate(pdf.pages):
                # Try lattice first (bordered tables), then stream (borderless)
                for strategy, settings in [
                    ("lattice", {
                        "vertical_strategy":   "lines_strict",
                        "horizontal_strategy": "lines_strict",
                        "snap_tolerance": 3, "join_tolerance": 3,
                        "edge_min_length": 3,
                    }),
                    ("stream", {
                        "vertical_strategy":   "text",
                        "horizontal_strategy": "text",
                        "snap_tolerance": 5, "join_tolerance": 5,
                        "min_words_vertical": 2, "min_words_horizontal": 2,
                    }),
                ]:
                    try:
                        tables = page.extract_tables(settings) or []
                        for tbl in tables:
                            if not tbl or len(tbl) < 2 or len(tbl[0]) < 2:
                                continue
                            header  = [str(c).strip() if c else "" for c in tbl[0]]
                            cleaned = [[str(c).strip() if c else "" for c in row] for row in tbl]
                            # Skip tables that are all-empty
                            non_empty = sum(1 for r in cleaned for c in r if c)
                            if non_empty < 4:
                                continue
                            rows_data = [header] + cleaned[1:]
                            csv_lines = [",".join(f'"{c}"' for c in row) for row in rows_data]
                            caption   = _find_table_caption(pdf_path, pnum)
                            results.append({
                                "csv":     f"[Table p{pnum+1}]" +
                                           (f" {caption}" if caption else "") +
                                           "\n" + "\n".join(csv_lines),
                                "header":  header,
                                "page":    pnum + 1,
                                "rows":    len(cleaned),
                                "cols":    len(cleaned[0]),
                                "caption": caption,
                                "source":  f"pdfplumber_{strategy}",
                            })
                            console.print(
                                f"  [dim]pdfplumber {strategy} table "
                                f"p{pnum+1}: {len(cleaned)}×{len(cleaned[0])}[/dim]"
                            )
                        if results:
                            break   # lattice found tables — skip stream
                    except Exception:
                        continue
    except ImportError:
        console.print("  [yellow]pdfplumber not installed[/yellow]")
    except Exception as e:
        console.print(f"  [yellow]pdfplumber failed: {e}[/yellow]")
    return results


def _pymupdf_text(pdf_path: str) -> str:
    """PyMuPDF layout-aware text extraction with junk stripping."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        parts = []
        for page in doc:
            parts.append(page.get_text("text", sort=True))
        doc.close()
        text = "\n".join(parts)
        return _strip_junk_sections(clean_text(text))
    except ImportError:
        console.print("  [yellow]PyMuPDF not installed[/yellow]")
        return ""
    except Exception as e:
        console.print(f"  [yellow]PyMuPDF text failed: {e}[/yellow]")
        return ""


def _pymupdf_images(pdf_path: str, output_dir: str) -> list[dict]:
    """PyMuPDF image extraction with size filtering."""
    results: list[dict] = []
    min_area = cfg.MIN_IMAGE_AREA_PX
    try:
        import fitz
        doc = fitz.open(pdf_path)
        for i in range(len(doc)):
            page_text = doc[i].get_text()
            fig_captions = re.findall(
                r"(Fig(?:ure)?\s+\d+[.:]\s*[^\n]{5,200})", page_text, re.I
            )
            for img_idx, img in enumerate(doc.get_page_images(i)):
                xref   = img[0]
                width  = img[2]
                height = img[3]
                if width * height < min_area:
                    continue
                base_image  = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext   = base_image.get("ext", "png")
                if image_ext.lower() in ("jbig2", "jpx"):
                    image_ext = "png"
                fname   = f"page_{i+1}_img_{img_idx+1}.{image_ext}"
                fpath   = os.path.join(output_dir, fname)
                with open(fpath, "wb") as fh:
                    fh.write(image_bytes)
                caption = fig_captions[img_idx] if img_idx < len(fig_captions) else ""
                results.append({
                    "path":    fpath,
                    "page":    i + 1,
                    "caption": caption.strip(),
                    "width":   width,
                    "height":  height,
                })
        doc.close()
    except ImportError:
        console.print("  [yellow]PyMuPDF not installed[/yellow]")
    except Exception as e:
        console.print(f"  [yellow]PyMuPDF image extraction failed: {e}[/yellow]")
    return results


def _pytesseract_ocr(pdf_path: str) -> str:
    """Tesseract OCR — last-resort for fully scanned PDFs."""
    console.print("  [yellow]pytesseract OCR fallback…[/yellow]")
    try:
        import numpy as np
        import cv2
        import pytesseract
        from pdf2image import convert_from_path

        pytesseract.pytesseract.tesseract_cmd = cfg.TESSERACT_CMD
        images = convert_from_path(pdf_path, dpi=300)
        parts  = []
        for i, img in enumerate(images):
            console.print(f"    [dim]OCR page {i+1}/{len(images)}[/dim]")
            arr  = np.array(img)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            parts.append(pytesseract.image_to_string(thresh, lang="eng"))
        return "\n\n".join(parts)
    except Exception as e:
        console.print(f"  [red]Tesseract OCR failed: {e}[/red]")
        return ""


def _count_pdf_pages(pdf_path: str) -> int:
    """Best-effort page count without full parse."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return 0


def _fallback_extract_all(pdf_path: str, image_dir: str) -> DoclingResult:
    """Full extraction using only PyMuPDF + pdfplumber when Docling is absent."""
    r = DoclingResult(method="fallback")
    r.full_text   = _pymupdf_text(pdf_path)
    r.text_blocks = chunk_text(r.full_text) if r.full_text else []
    r.tables      = _pdfplumber_tables(pdf_path)
    r.figures     = _pymupdf_images(pdf_path, image_dir) if image_dir else []
    r.page_count  = _count_pdf_pages(pdf_path)
    return r


# ── Helper: table caption from PyMuPDF ────────────────────────────────────────

def _find_table_caption(pdf_path: str, page_num: int) -> str:
    """Find a 'Table N.' caption on the given page using PyMuPDF."""
    try:
        import fitz
        doc  = fitz.open(pdf_path)
        page = doc[page_num]
        text = page.get_text()
        doc.close()
        m = re.search(r"(Table\s+\d+[.:]\s*[^\n]{5,120})", text, re.I)
        return m.group(1).strip() if m else ""
    except Exception:
        return ""


# ── Text cleaning and chunking ────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean extracted text: strip citation noise, normalise whitespace."""
    # Fix broken bracket citations
    text = re.sub(r"\[\s*\n\s*([A-Za-z0-9]+)\s*\n\s*\]", r"[\1]", text)
    text = re.sub(r"\[\s*\n\s*\]", "", text)
    # Strip numeric citations [1], [1,2], [1-3]
    text = re.sub(r"\[\s*\d+(?:[\s,\-]*\d+)*\s*\]", "", text)
    # Strip et al. references
    text = re.sub(r"\(\s*[A-Za-z]+ et al\.,?\s*\d{4}\s*\)", "", text)
    # Strip bare URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Strip DOI inline markers
    text = re.sub(
        r"\[\s*(?:DOI|PMC free article|PubMed|Google Scholar|Open in a new tab)\s*\]",
        "", text, flags=re.I,
    )
    text = re.sub(r"doi:\s*10\.\S+", "", text, flags=re.I)
    # Strip lone page numbers
    lines, cleaned = text.split("\n"), []
    for line in lines:
        s = line.strip()
        if len(s) < 3 and s.isdigit():
            continue
        cleaned.append(line)
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    max_chars: int | None = None,
    overlap: int = 500,
) -> list[str]:
    """
    Section-aware text chunking with table-block protection.

    Tables are kept atomic — a chunk boundary is never placed inside a
    detected table block (lines starting with | or consistent CSV structure).
    """
    if max_chars is None:
        max_chars = cfg.MAX_TEXT_CHARS
    text = clean_text(text)
    if len(text) <= max_chars:
        return [text]

    section_pattern = re.compile(
        r"\n(?="
        r"(?:\d{1,3}(?:\.\d{1,3})*\.?\s+[A-Z])"
        r"|(?:[A-Z][A-Z\s]{4,}(?:\n|$))"
        r"|(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\n[-=]{3,})"
        r"|(?:#{1,4}\s+\S)"
        r")"
    )

    section_starts = [0] + [m.start() + 1 for m in section_pattern.finditer(text)]
    sections: list[str] = []
    for i, start in enumerate(section_starts):
        end = section_starts[i + 1] if i + 1 < len(section_starts) else len(text)
        sec = text[start:end].strip()
        if sec:
            sections.append(sec)

    chunks: list[str] = []
    current_chunk = ""

    for section in sections:
        if current_chunk and len(current_chunk) + len(section) + 2 > max_chars:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = ""

        if len(section) > max_chars:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Split large section, protecting table blocks
            pos = 0
            while pos < len(section):
                end = pos + max_chars
                sub = section[pos:end]
                if end < len(section):
                    # Avoid splitting inside a table block
                    if _is_inside_table_block(sub):
                        # Walk back to before the table starts
                        tbl_start = _find_table_block_start(sub)
                        if tbl_start > max_chars * 0.3:
                            sub = sub[:tbl_start]
                            end = pos + tbl_start
                    else:
                        para_break = sub.rfind("\n\n")
                        if para_break > max_chars * 0.4:
                            sub = sub[:para_break]
                            end = pos + para_break
                        else:
                            sent_break = sub.rfind(". ")
                            if sent_break > max_chars * 0.4:
                                sub = sub[:sent_break + 1]
                                end = pos + sent_break + 1
                if sub.strip():
                    chunks.append(sub.strip())
                pos = end - overlap if end < len(section) else end
        else:
            if current_chunk:
                current_chunk += "\n\n" + section
            else:
                current_chunk = section

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text[:max_chars]]


def _is_inside_table_block(text: str) -> bool:
    """Return True if the tail of text appears to be inside a table."""
    tail = text[-500:] if len(text) > 500 else text
    pipe_lines = sum(1 for l in tail.splitlines() if "|" in l)
    return pipe_lines >= 3


def _find_table_block_start(text: str) -> int:
    """Return the char index where the trailing table block starts."""
    lines  = text.splitlines(keepends=True)
    pos    = len(text)
    in_tbl = False
    for line in reversed(lines):
        if "|" in line:
            pos -= len(line)
            in_tbl = True
        elif in_tbl:
            break
        else:
            pos -= len(line)
    return max(0, pos)