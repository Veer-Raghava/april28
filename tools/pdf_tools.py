"""
tools/pdf_tools.py — PDF extraction using Docling as primary engine.

Docling handles:
  - Digital PDFs (text layer): geometric cell detection, orientation-independent
  - Scanned / rotated PDFs: EasyOCR angle detection → TableFormer structure recovery
  - Horizontal / sideways tables (90°/270°): rasterises page, runs OCR, reconstructs table
  - Borderless academic tables: TableFormer ACCURATE mode (deep learning)
  - Multi-column layouts: Docling layout model parses reading order correctly

Cascade:
  1. Docling (primary) — TableFormer ACCURATE + optional EasyOCR
  2. pdfplumber (lattice + stream) — fallback if Docling finds 0 tables
  3. PyMuPDF — text and image fallback
  4. pytesseract — last-resort OCR for fully scanned pages

Install:
    pip install docling docling-core
    pip install pdfplumber pymupdf  # kept as fallbacks
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.console_setup import console
import config as cfg


# ── Duplicate detection ───────────────────────────────────────────────────────
_seen_hashes: set[str] = set()


def is_duplicate_pdf(pdf_path: str) -> bool:
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
        "    pip install docling docling-core[/yellow]"
    )


# ── Junk section patterns ─────────────────────────────────────────────────────
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
    """Truncate text at first junk section (References, Acknowledgements, etc.)."""
    match = _JUNK_SECTION_RE.search(text)
    return text[:match.start()].strip() if match else text


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class DoclingResult:
    full_text:   str            = ""
    text_blocks: list[str]      = field(default_factory=list)
    tables:      list[dict]     = field(default_factory=list)
    figures:     list[dict]     = field(default_factory=list)
    page_count:  int            = 0
    method:      str            = "unknown"
    warnings:    list[str]      = field(default_factory=list)


# ── Docling pipeline builder ──────────────────────────────────────────────────
def _build_docling_converter(do_ocr: bool = False,
                              force_full_page_ocr: bool = False):
    opts = PdfPipelineOptions()
    # ACCURATE mode: deep-learning TableFormer — handles rotated, borderless,
    # merged-cell, and complex academic tables
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.table_structure_options.do_cell_matching = True
    opts.do_ocr = do_ocr

    if do_ocr:
        try:
            import easyocr  # noqa
            # EasyOCR has built-in angle detection — handles rotated text
            opts.ocr_options = EasyOcrOptions(
                force_full_page_ocr=force_full_page_ocr,
                lang=["en"],
            )
            console.print("  [dim]OCR: EasyOCR (rotation-aware)[/dim]")
        except ImportError:
            opts.ocr_options = TesseractCliOcrOptions(
                force_full_page_ocr=force_full_page_ocr,
                lang="eng",
                tesseract_cmd=cfg.TESSERACT_CMD,
            )
            console.print("  [dim]OCR: Tesseract CLI[/dim]")

    # Docling option names vary between versions.
    # Keep compatibility by setting attributes only if present.
    try:
        if hasattr(opts, "force_full_page_ocr"):
            setattr(opts, "force_full_page_ocr", force_full_page_ocr)
    except Exception:
        pass

    opts.generate_picture_images = True
    opts.images_scale = 2.0

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


# ── Single-pass full extraction ───────────────────────────────────────────────
def extract_all(
    pdf_path: str,
    image_dir: str = "",
    do_ocr: bool | None = None,
    force_full_page_ocr: bool = False,
) -> DoclingResult:
    """
    Full extraction in one Docling pass.

    Rotated/horizontal table strategy
    ----------------------------------
    Digital PDFs (text layer present):
      Docling's geometric PDF backend detects cell boundaries from vector
      graphics regardless of page rotation. 90° rotated tables work correctly.

    Scanned / image PDFs (or digital with embedded rotated tables):
      With do_ocr=True, Docling rasterises pages at 2× scale, EasyOCR
      detects text at any angle, and TableFormer reconstructs the structure.
      Set force_full_page_ocr=True for mixed digital+rotated documents.
    """
    if not _DOCLING_AVAILABLE:
        return _fallback_extract_all(pdf_path, image_dir)

    if do_ocr is None:
        do_ocr = cfg.ENABLE_OCR

    result = DoclingResult(method="docling+ocr" if do_ocr else "docling")
    passes: list[tuple[bool, bool]] = [(do_ocr, force_full_page_ocr)]
    if cfg.ENABLE_OCR and not do_ocr:
        passes.append((True, False))
    if cfg.ENABLE_OCR and (True, True) not in passes:
        passes.append((True, True))

    last_error = ""
    for pass_idx, (pass_ocr, pass_force) in enumerate(passes, start=1):
        try:
            converter = _build_docling_converter(
                do_ocr=pass_ocr,
                force_full_page_ocr=pass_force,
            )
            console.print(
                f"  [dim]📄 Docling pass {pass_idx}/{len(passes)} "
                f"(ocr={pass_ocr}, full_ocr={pass_force}) for {Path(pdf_path).name}…[/dim]"
            )
            t0 = time.time()
            conv = converter.convert(pdf_path)
            doc = conv.document
            console.print(f"  [dim]Docling done in {time.time()-t0:.1f}s[/dim]")

            result.page_count = getattr(doc, "num_pages", 0) or _count_pdf_pages(pdf_path)
            result.method = "docling+ocr" if pass_ocr else "docling"
            raw_text_parts: list[str] = []
            seen_text: set[str] = set()
            tbl_idx = len(result.tables)

            for item, _ in doc.iterate_items():
                label = getattr(item, "label", None)

                if label in (DocItemLabel.PARAGRAPH, DocItemLabel.TEXT,
                             DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE,
                             DocItemLabel.CAPTION, DocItemLabel.LIST_ITEM):
                    text = clean_text(getattr(item, "text", "").strip())
                    if not text:
                        continue
                    if text.lower() in _JUNK_HEADERS_SET:
                        break
                    if text in seen_text:
                        continue
                    seen_text.add(text)
                    raw_text_parts.append(text)

                elif label == DocItemLabel.TABLE:
                    tbl = _extract_docling_table(item, doc, tbl_idx, pdf_path)
                    if tbl:
                        result.tables.append(tbl)
                        tbl_idx += 1
                        console.print(
                            f"  [dim]📊 Table {tbl_idx}: {tbl['rows']}×{tbl['cols']}"
                            f" p{tbl['page']} [{tbl.get('source','?')}][/dim]"
                        )

                elif label == DocItemLabel.PICTURE:
                    fig = _extract_docling_figure(item, doc, image_dir, len(result.figures))
                    if fig:
                        result.figures.append(fig)

            if not raw_text_parts:
                raw_text_parts = _collect_docling_text_fallback(doc)

            raw_text = "\n\n".join(raw_text_parts)
            if raw_text.strip():
                result.full_text = _strip_junk_sections(clean_text(raw_text))
                result.text_blocks = chunk_text(result.full_text) if result.full_text else []

            if result.full_text and result.tables:
                return result
            if result.full_text and pass_ocr:
                return result

        except Exception as e:
            last_error = str(e)
            console.print(f"  [yellow]⚠ Docling pass {pass_idx} failed: {e}[/yellow]")

    # Fallback: 0 tables → try pdfplumber
    if not result.tables:
        console.print("  [dim yellow]Docling: 0 tables — trying pdfplumber fallback…[/dim yellow]")
        fb = _pdfplumber_tables(pdf_path)
        if fb:
            result.tables = fb
            result.warnings.append("tables_via_pdfplumber_fallback")

    # Fallback: no text → PyMuPDF
    if not result.full_text:
        console.print("  [dim yellow]Docling: no text — trying PyMuPDF…[/dim yellow]")
        result.full_text = _pymupdf_text(pdf_path)
        result.text_blocks = chunk_text(result.full_text) if result.full_text else []
        result.warnings.append("text_via_pymupdf_fallback")

    if last_error:
        result.warnings.append(f"docling_failed:{last_error}")
    return result


def _collect_docling_text_fallback(doc: Any) -> list[str]:
    """Try multiple document-level exporters to recover text if item walk is sparse."""
    parts: list[str] = []
    for method in ("export_to_markdown", "export_to_text"):
        fn = getattr(doc, method, None)
        if not callable(fn):
            continue
        try:
            dumped = fn()
            if isinstance(dumped, str) and dumped.strip():
                parts.extend(p.strip() for p in dumped.split("\n\n") if p.strip())
                if parts:
                    break
        except Exception:
            continue
    return parts


# ── Docling table extractor ───────────────────────────────────────────────────
def _extract_docling_table(item, doc, index: int, pdf_path: str) -> dict | None:
    try:
        df = item.export_to_dataframe(doc)
    except Exception:
        try:
            md = item.export_to_markdown(doc)
            df = _markdown_table_to_df(md)
        except Exception:
            return None

    if df is None or df.shape[0] < 1 or df.shape[1] < 2:
        return None

    import pandas as pd
    df = df.fillna("").astype(str).applymap(str.strip)
    df = df.loc[~(df == "").all(axis=1)]
    df = df.loc[:, ~(df == "").all(axis=0)]
    if df.shape[0] < 1 or df.shape[1] < 2:
        return None

    header = list(df.columns)
    rows_data = [header] + df.values.tolist()
    csv_lines = [",".join(f'"{str(c)}"' for c in row) for row in rows_data]

    caption = _find_docling_caption(item, doc, index)
    if not caption:
        caption = _find_table_caption(pdf_path,
                                       getattr(item, "page_no", 0) or 0)
    page_no = 0
    try:
        page_no = item.prov[0].page_no if item.prov else 0
    except Exception:
        pass

    return {
        "csv":     f"[Table {index+1} p{page_no}]" +
                   (f" {caption}" if caption else "") +
                   "\n" + "\n".join(csv_lines),
        "header":  header,
        "page":    page_no,
        "rows":    df.shape[0],
        "cols":    df.shape[1],
        "caption": caption,
        "source":  "docling",
        "df":      df,
    }


def _find_docling_caption(item, doc, index: int) -> str:
    try:
        if hasattr(item, "caption_text") and callable(item.caption_text):
            return item.caption_text(doc).strip()
        cap = getattr(item, "caption", None)
        if cap:
            return str(cap.text if hasattr(cap, "text") else cap).strip()
    except Exception:
        pass
    return ""


def _markdown_table_to_df(md: str):
    try:
        import pandas as pd
        lines = [l.strip() for l in md.strip().splitlines()
                 if l.strip().startswith("|")]
        lines = [l for l in lines if not re.match(r"^\|[\s\-:|]+\|$", l)]
        rows = [[c.strip() for c in l.strip("|").split("|")] for l in lines]
        return pd.DataFrame(rows[1:], columns=rows[0]) if len(rows) >= 2 else None
    except Exception:
        return None


# ── Docling figure extractor ──────────────────────────────────────────────────
def _extract_docling_figure(item, doc, image_dir: str, index: int) -> dict | None:
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
        fig: dict = {"page": page_no, "caption": caption, "width": w,
                     "height": h, "path": ""}
        if image_dir:
            os.makedirs(image_dir, exist_ok=True)
            fpath = os.path.join(image_dir, f"figure_{index+1}_p{page_no}.png")
            pil_img.save(fpath, format="PNG")
            fig["path"] = fpath
        return fig
    except Exception:
        return None


# ── Public backward-compatible API ───────────────────────────────────────────

def extract_text(pdf_path: str) -> str:
    """Extract body text. Uses Docling → PyMuPDF. Skips duplicates."""
    if is_duplicate_pdf(pdf_path):
        return ""
    if _DOCLING_AVAILABLE:
        try:
            r = extract_all(pdf_path, image_dir="", do_ocr=False)
            if r.full_text and len(r.full_text.strip()) >= 300:
                return r.full_text
            if cfg.ENABLE_OCR:
                console.print("  [dim yellow]Short text — retrying with OCR…[/dim yellow]")
                r2 = extract_all(pdf_path, image_dir="", do_ocr=True)
                if r2.full_text:
                    return r2.full_text
        except Exception as e:
            console.print(f"  [dim yellow]Docling text failed: {e}[/dim yellow]")
    return _pymupdf_text(pdf_path)


def extract_text_ocr(pdf_path: str) -> str:
    """OCR-based extraction for fully scanned PDFs."""
    if not cfg.ENABLE_OCR:
        return ""
    console.print("  [yellow]OCR extraction starting…[/yellow]")
    if _DOCLING_AVAILABLE:
        try:
            r = extract_all(pdf_path, image_dir="", do_ocr=True,
                            force_full_page_ocr=True)
            if r.full_text and len(r.full_text.strip()) >= 100:
                return r.full_text
        except Exception as e:
            console.print(f"  [dim yellow]Docling OCR failed: {e}[/dim yellow]")
    return _pytesseract_ocr(pdf_path)


def extract_tables(pdf_path: str) -> list[dict]:
    """
    Extract tables. Docling TableFormer → pdfplumber fallback.
    Docling handles: rotated, borderless, merged-cell, multi-header tables.
    """
    if _DOCLING_AVAILABLE:
        try:
            r = extract_all(pdf_path, image_dir="", do_ocr=False)
            if r.tables:
                return r.tables
            # No tables without OCR — retry with OCR (catches embedded rotated tables)
            if cfg.ENABLE_OCR:
                console.print(
                    "  [dim yellow]No tables without OCR — retrying with OCR "
                    "(handles rotated/scanned tables)…[/dim yellow]"
                )
                r2 = extract_all(pdf_path, image_dir="", do_ocr=True,
                                 force_full_page_ocr=False)
                if r2.tables:
                    return r2.tables
        except Exception as e:
            console.print(f"  [dim yellow]Docling table extraction failed: {e}[/dim yellow]")
    return _pdfplumber_tables(pdf_path)


def extract_images(pdf_path: str, output_dir: str) -> list[dict]:
    """Extract figures, filtering out small/irrelevant images."""
    os.makedirs(output_dir, exist_ok=True)
    if _DOCLING_AVAILABLE:
        try:
            r = extract_all(pdf_path, image_dir=output_dir, do_ocr=False)
            if r.figures:
                return r.figures
        except Exception as e:
            console.print(f"  [dim yellow]Docling image extraction failed: {e}[/dim yellow]")
    return _pymupdf_images(pdf_path, output_dir)


# ── Internal fallbacks ────────────────────────────────────────────────────────

def _pdfplumber_tables(pdf_path: str) -> list[dict]:
    """pdfplumber table extraction: lattice first, then stream."""
    results: list[dict] = []
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for pnum, page in enumerate(pdf.pages):
                for strategy, settings in [
                    ("lattice", {
                        "vertical_strategy": "lines_strict",
                        "horizontal_strategy": "lines_strict",
                        "snap_tolerance": 3, "join_tolerance": 3,
                        "edge_min_length": 3,
                    }),
                    ("stream", {
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "snap_tolerance": 5, "join_tolerance": 5,
                        "min_words_vertical": 2, "min_words_horizontal": 2,
                    }),
                ]:
                    for tbl in page.extract_tables(settings) or []:
                        if not tbl or len(tbl) < 2 or len(tbl[0]) < 2:
                            continue
                        header = [str(c).strip() if c else "" for c in tbl[0]]
                        cleaned = [[str(c).strip() if c else "" for c in row]
                                   for row in tbl]
                        if sum(1 for r in cleaned for c in r if c) < 4:
                            continue
                        rows_data = [header] + cleaned[1:]
                        csv_lines = [",".join(f'"{c}"' for c in row)
                                     for row in rows_data]
                        caption = _find_table_caption(pdf_path, pnum)
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
                        break
    except ImportError:
        console.print("  [yellow]pdfplumber not installed[/yellow]")
    except Exception as e:
        console.print(f"  [yellow]pdfplumber failed: {e}[/yellow]")
    return results


def _pymupdf_text(pdf_path: str) -> str:
    try:
        import fitz
        doc = fitz.open(pdf_path)
        parts = [page.get_text("text", sort=True) for page in doc]
        doc.close()
        return _strip_junk_sections(clean_text("\n".join(parts)))
    except ImportError:
        return ""
    except Exception as e:
        console.print(f"  [yellow]PyMuPDF text failed: {e}[/yellow]")
        return ""


def _pymupdf_images(pdf_path: str, output_dir: str) -> list[dict]:
    results: list[dict] = []
    min_area = cfg.MIN_IMAGE_AREA_PX
    try:
        import fitz
        doc = fitz.open(pdf_path)
        for i in range(len(doc)):
            page_text  = doc[i].get_text()
            fig_caps   = re.findall(r"(Fig(?:ure)?\s+\d+[.:]\s*[^\n]{5,200})",
                                    page_text, re.I)
            for img_idx, img in enumerate(doc.get_page_images(i)):
                xref, w, h = img[0], img[2], img[3]
                if w * h < min_area:
                    continue
                base_image  = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext   = base_image.get("ext", "png")
                if image_ext.lower() in ("jbig2", "jpx"):
                    image_ext = "png"
                fname = f"page_{i+1}_img_{img_idx+1}.{image_ext}"
                fpath = os.path.join(output_dir, fname)
                with open(fpath, "wb") as fh:
                    fh.write(image_bytes)
                caption = fig_caps[img_idx] if img_idx < len(fig_caps) else ""
                results.append({"path": fpath, "page": i+1,
                                 "caption": caption.strip(),
                                 "width": w, "height": h})
        doc.close()
    except ImportError:
        pass
    except Exception as e:
        console.print(f"  [yellow]PyMuPDF image extraction failed: {e}[/yellow]")
    return results


def _pytesseract_ocr(pdf_path: str) -> str:
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
            arr   = np.array(img)
            gray  = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            _, th = cv2.threshold(gray, 150, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            parts.append(pytesseract.image_to_string(th, lang="eng"))
        return "\n\n".join(parts)
    except Exception as e:
        console.print(f"  [red]Tesseract OCR failed: {e}[/red]")
        return ""


def _count_pdf_pages(pdf_path: str) -> int:
    try:
        import fitz
        doc = fitz.open(pdf_path)
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return 0


def _fallback_extract_all(pdf_path: str, image_dir: str) -> DoclingResult:
    r = DoclingResult(method="fallback")
    r.full_text   = _pymupdf_text(pdf_path)
    r.text_blocks = chunk_text(r.full_text) if r.full_text else []
    r.tables      = _pdfplumber_tables(pdf_path)
    r.figures     = _pymupdf_images(pdf_path, image_dir) if image_dir else []
    r.page_count  = _count_pdf_pages(pdf_path)
    return r


def _find_table_caption(pdf_path: str, page_num: int) -> str:
    try:
        import fitz
        doc  = fitz.open(pdf_path)
        text = doc[page_num].get_text()
        doc.close()
        m = re.search(r"(Table\s+\d+[.:]\s*[^\n]{5,120})", text, re.I)
        return m.group(1).strip() if m else ""
    except Exception:
        return ""


# ── Text cleaning and chunking ────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean extracted text: strip citations, normalise whitespace."""
    text = re.sub(r'\[\s*\n\s*([A-Za-z0-9]+)\s*\n\s*\]', r'[\1]', text)
    text = re.sub(r'\[\s*\n\s*\]', '', text)
    text = re.sub(r'\[\s*\d+(?:[\s,\-]*\d+)*\s*\]', '', text)
    text = re.sub(r'\(\s*[A-Za-z]+ et al\.,?\s*\d{4}\s*\)', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(
        r'\[\s*(?:DOI|PMC free article|PubMed|Google Scholar|Open in a new tab)\s*\]',
        '', text, flags=re.I)
    text = re.sub(r'doi:\s*10\.\S+', '', text, flags=re.I)
    lines, cleaned = text.split('\n'), []
    for line in lines:
        s = line.strip()
        if len(s) < 3 and s.isdigit():
            continue
        cleaned.append(line)
    text = '\n'.join(cleaned)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def chunk_text(text: str, max_chars: int | None = None,
               overlap: int = 500) -> list[str]:
    """
    Section-aware chunking. Table blocks are never split across chunks.
    """
    if max_chars is None:
        max_chars = cfg.MAX_TEXT_CHARS
    text = clean_text(text)
    if len(text) <= max_chars:
        return [text]

    section_pattern = re.compile(
        r'\n(?='
        r'(?:\d{1,3}(?:\.\d{1,3})*\.?\s+[A-Z])'
        r'|(?:[A-Z][A-Z\s]{4,}(?:\n|$))'
        r'|(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\n[-=]{3,})'
        r'|(?:#{1,4}\s+\S)'
        r')'
    )
    section_starts = [0] + [m.start() + 1 for m in section_pattern.finditer(text)]
    sections: list[str] = []
    for i, start in enumerate(section_starts):
        end = section_starts[i+1] if i+1 < len(section_starts) else len(text)
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
            pos = 0
            while pos < len(section):
                end = pos + max_chars
                sub = section[pos:end]
                if end < len(section):
                    # Don't split inside a table block
                    if _is_inside_table_block(sub):
                        tbl_start = _find_table_block_start(sub)
                        if tbl_start > max_chars * 0.3:
                            sub = sub[:tbl_start]
                            end = pos + tbl_start
                    else:
                        pb = sub.rfind("\n\n")
                        if pb > max_chars * 0.4:
                            sub = sub[:pb]; end = pos + pb
                        else:
                            sb = sub.rfind(". ")
                            if sb > max_chars * 0.4:
                                sub = sub[:sb+1]; end = pos + sb + 1
                if sub.strip():
                    chunks.append(sub.strip())
                pos = end - overlap if end < len(section) else end
        else:
            current_chunk = (current_chunk + "\n\n" + section
                             if current_chunk else section)

    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks if chunks else [text[:max_chars]]


def _is_inside_table_block(text: str) -> bool:
    tail = text[-500:] if len(text) > 500 else text
    return sum(1 for l in tail.splitlines() if "|" in l) >= 3


def _find_table_block_start(text: str) -> int:
    lines = text.splitlines(keepends=True)
    pos = len(text)
    in_tbl = False
    for line in reversed(lines):
        if "|" in line:
            pos -= len(line); in_tbl = True
        elif in_tbl:
            break
        else:
            pos -= len(line)
    return max(0, pos)
