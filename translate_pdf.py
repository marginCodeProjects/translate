#!/usr/bin/env python3
"""
PDF Double Translation Tool
============================
Extracts text from a PDF, translates each paragraph via a local LLM
(vLLM with OpenAI-compatible API), and builds a new PDF with interleaved
original (English) and translated (Russian) paragraphs.

Usage:
    python translate_pdf.py \
        --input book.pdf \
        --output book_translated.pdf \
        --api-url http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-72B-Instruct
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import fitz  # PyMuPDF
from fpdf import FPDF
from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FONT_DIR = Path(__file__).parent / "fonts"
IMAGE_DIR = Path(__file__).parent / "images"
DEJAVU_ZIP_URL = (
    "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/"
    "version_2_37/dejavu-fonts-ttf-2.37.zip"
)

SYSTEM_PROMPT = """\
You are a professional literary translator from English to Russian.
Translate the following text accurately and naturally into Russian,
preserving the author's style, tone, and meaning.
Do NOT add any commentary, notes, or explanations — output ONLY the translation.
If the text contains proper nouns, transliterate them into Russian.\
"""

# Minimum paragraph length (characters) to bother translating
MIN_PARAGRAPH_LEN = 20

# How many paragraphs to batch into a single LLM request
BATCH_SIZE = 5

# Max retries on API errors
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Font management
# ---------------------------------------------------------------------------


def ensure_fonts() -> tuple[Path, Path]:
    """Download DejaVu Sans (regular + bold) if not already cached."""
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    regular = FONT_DIR / "DejaVuSans.ttf"
    bold = FONT_DIR / "DejaVuSans-Bold.ttf"

    if not regular.exists() or not bold.exists():
        print("Downloading DejaVu fonts ...")
        resp = urllib.request.urlopen(DEJAVU_ZIP_URL)
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
        for member in archive.namelist():
            basename = os.path.basename(member)
            if basename == "DejaVuSans.ttf":
                regular.write_bytes(archive.read(member))
                print(f"  -> {regular}")
            elif basename == "DejaVuSans-Bold.ttf":
                bold.write_bytes(archive.read(member))
                print(f"  -> {bold}")
        archive.close()

    return regular, bold


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------


def _block_to_text(block: dict) -> str:
    """Concatenate all spans in a text block into a single string."""
    parts: list[str] = []
    for line in block["lines"]:
        line_text = "".join(span["text"] for span in line["spans"])
        parts.append(line_text)
    raw = " ".join(parts).strip()
    return re.sub(r"\s+", " ", raw)


def _block_has_bold(block: dict) -> bool:
    """Check if any span in the block uses a bold font."""
    for line in block["lines"]:
        for span in line["spans"]:
            if "Bold" in span.get("font", ""):
                return True
    return False


def _block_line_count(block: dict) -> int:
    """Number of lines in a block."""
    return len(block.get("lines", []))


def _is_chapter_number(text: str) -> bool:
    """Check if text is just a chapter number like '01', '02', ... '50'."""
    return bool(re.fullmatch(r"\d{1,2}", text.strip()))


_SENTENCE_END = re.compile(r'[.!?"\u201D\u2019]\s*$')


def _ends_sentence(text: str) -> bool:
    """Check if text ends with sentence-ending punctuation."""
    return bool(_SENTENCE_END.search(text))


# Minimum image coverage (fraction of page area) to be considered a real photo.
# Tiny decorative icons (bullets etc.) are ~0.05% and should be skipped.
MIN_IMAGE_COVERAGE = 0.05

# Pages to skip: front matter (title, copyright, intro, TOC)
# Pages 1-5: empty/cover, 6-7: copyright, 7-9: intro, 9-11: TOC
CONTENT_START_PAGE = 12  # First chapter number page

# Pages containing the Table of Contents
TOC_PAGES = range(8, 12)  # 0-based: pages 9-12


def _extract_toc(doc) -> dict[int, str]:
    """
    Extract the table of contents from the PDF.
    Returns {chapter_number: title} for all 50 chapters.
    """
    toc: dict[int, str] = {}
    for pn in TOC_PAGES:
        if pn >= len(doc):
            break
        page = doc[pn]
        data = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        for block in data["blocks"]:
            if block["type"] != 0:
                continue
            spans = []
            for line in block["lines"]:
                for span in line["spans"]:
                    spans.append(span)
            # TOC entry = bold number span + regular title span(s)
            if len(spans) >= 2 and "Bold" in spans[0].get("font", ""):
                num_text = spans[0]["text"].strip()
                if re.fullmatch(r"\d{1,2}", num_text):
                    ch = int(num_text)
                    title = "".join(s["text"] for s in spans[1:]).strip()
                    title = re.sub(r"\s+", " ", title)
                    toc[ch] = title
    return toc


def _extract_page_images(doc, page_num: int) -> list[dict]:
    """
    Extract significant images from a page.
    Returns list of dicts: {"xref": int, "rect": fitz.Rect, "coverage": float}
    Skips tiny decorative images.
    """
    page = doc[page_num]
    pw, ph = page.rect.width, page.rect.height
    page_area = pw * ph
    results = []
    seen_xrefs: set[int] = set()

    for img_info in page.get_images(full=True):
        xref = img_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        for rect in page.get_image_rects(xref):
            coverage = (rect.width * rect.height) / page_area if page_area else 0
            if coverage >= MIN_IMAGE_COVERAGE:
                results.append({
                    "xref": xref,
                    "rect": rect,
                    "coverage": coverage,
                })
    return results


def _save_image(doc, xref: int, out_dir: Path) -> Path | None:
    """Extract an image by xref and save to disk. Returns the file path."""
    try:
        img = doc.extract_image(xref)
        if not img or not img.get("image"):
            return None
        ext = img.get("ext", "png")
        path = out_dir / f"img_{xref}.{ext}"
        if not path.exists():
            path.write_bytes(img["image"])
        return path
    except Exception:
        return None


def extract_paragraphs(pdf_path: str) -> list[dict]:
    """
    Extract structured text and images from the PDF.

    Returns a list of dicts. Each dict has a "kind" field:
      - "chapter_title": {"text", "chapter", "page", "kind"}
      - "body" / "caption": {"text", "chapter", "page", "kind"}
      - "image": {"image_path", "width", "height", "page", "kind", "chapter"}

    Images are inserted at their page position relative to text blocks.
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    # --- Extract Table of Contents (authoritative chapter titles) ---
    toc = _extract_toc(doc)

    # --- Pass 1: collect raw blocks per page ---
    page_blocks: list[list[dict]] = []
    for page_num in range(total_pages):
        page = doc[page_num]
        data = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        blocks = []
        for block in data["blocks"]:
            if block["type"] != 0:
                continue
            text = _block_to_text(block)
            if len(text) < 2:
                continue
            blocks.append({
                "text": text,
                "bold": _block_has_bold(block),
                "nlines": _block_line_count(block),
                "page": page_num + 1,
            })
        page_blocks.append(blocks)

    # --- Detect back-matter start ---
    back_matter_start = total_pages
    for pn in range(total_pages - 1, max(total_pages - 10, 0), -1):
        for b in page_blocks[pn]:
            if b["text"] in ("Document Outline", "Table of Contents"):
                back_matter_start = pn
                break

    # --- Identify "photo pages": very short text pages (caption only) ---
    photo_pages: set[int] = set()
    for pn in range(total_pages):
        blocks = page_blocks[pn]
        total_chars = sum(len(b["text"]) for b in blocks)
        total_lines = sum(b["nlines"] for b in blocks)
        if 0 < total_chars < 150 and total_lines <= 3 and pn + 1 >= CONTENT_START_PAGE:
            if not (len(blocks) == 1 and _is_chapter_number(blocks[0]["text"])):
                photo_pages.add(pn + 1)

    # --- Extract and save images ---
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    page_images: dict[int, list[dict]] = {}  # page_num (1-based) -> image info
    for pn in range(total_pages):
        if pn + 1 < CONTENT_START_PAGE or pn >= back_matter_start:
            continue
        imgs = _extract_page_images(doc, pn)
        if imgs:
            saved = []
            for img in imgs:
                path = _save_image(doc, img["xref"], IMAGE_DIR)
                if path:
                    saved.append({
                        "image_path": str(path),
                        "width": img["rect"].width,
                        "height": img["rect"].height,
                        "coverage": img["coverage"],
                        "y_pos": img["rect"].y0,  # vertical position on page
                        "page": pn + 1,
                    })
            if saved:
                page_images[pn + 1] = saved

    # --- Pass 2: flatten into linear sequence, skip front/back matter ---
    # Interleave text blocks and image entries based on page position.
    flat: list[dict] = []
    for pn in range(total_pages):
        if pn + 1 < CONTENT_START_PAGE or pn >= back_matter_start:
            continue

        # Collect text blocks for this page
        text_blocks = []
        for block in page_blocks[pn]:
            # Skip bold TOC entries on early pages ("01 From Where...", etc.)
            if block["bold"] and re.match(r"\d{2}\s", block["text"]) and pn + 1 <= 14:
                continue
            # Skip standalone "Contents" / "Introduction" labels
            if block["text"] in ("Contents", "Introduction") and pn + 1 <= 14:
                continue
            # Skip OceanofPDF markers
            if "OceanofPDF" in block["text"]:
                continue
            text_blocks.append(block)

        # Insert images for this page (before text if image is above text,
        # or between text blocks based on vertical position)
        imgs_on_page = page_images.get(pn + 1, [])
        if not imgs_on_page:
            flat.extend(text_blocks)
        else:
            # Simple strategy: insert each image before the first text block
            # that appears below it (by y-position). If no text blocks,
            # just add image entries.
            img_idx = 0
            imgs_sorted = sorted(imgs_on_page, key=lambda x: x["y_pos"])

            if not text_blocks:
                # Image-only page
                for img in imgs_sorted:
                    flat.append({
                        "text": "",
                        "kind": "_image",
                        "image_path": img["image_path"],
                        "img_width": img["width"],
                        "img_height": img["height"],
                        "page": pn + 1,
                        "bold": False,
                        "nlines": 0,
                        "_merge": False,
                    })
            else:
                # Interleave: add images before the text blocks they precede
                # For simplicity, add all page images before the first text block
                for img in imgs_sorted:
                    flat.append({
                        "text": "",
                        "kind": "_image",
                        "image_path": img["image_path"],
                        "img_width": img["width"],
                        "img_height": img["height"],
                        "page": pn + 1,
                        "bold": False,
                        "nlines": 0,
                        "_merge": False,
                    })
                flat.extend(text_blocks)

    # --- Pass 3: detect chapter numbers + titles, mark raw blocks ---
    tagged: list[dict] = []  # items with "kind" assigned
    i = 0
    current_chapter: int | None = None

    while i < len(flat):
        block = flat[i]

        # Pass through image entries as-is
        if block.get("kind") == "_image":
            tagged.append({
                "kind": "image",
                "image_path": block["image_path"],
                "img_width": block["img_width"],
                "img_height": block["img_height"],
                "chapter": current_chapter,
                "page": block["page"],
                "text": "",
                "_merge": False,
            })
            i += 1
            continue

        # Is this a chapter number block?
        if _is_chapter_number(block["text"]):
            ch_num = int(block["text"])
            current_chapter = ch_num
            i += 1

            # Use the authoritative title from the Table of Contents
            toc_title = toc.get(ch_num, f"Chapter {ch_num}")
            title_page = block["page"]
            # Find the next non-image block's page for accurate page info
            for j in range(i, min(i + 5, len(flat))):
                if flat[j].get("kind") != "_image":
                    title_page = flat[j]["page"]
                    break

            tagged.append({
                "text": toc_title,
                "kind": "chapter_title",
                "chapter": ch_num,
                "page": title_page,
                "_merge": False,
            })

            # Skip blocks that are part of the title text (they duplicate
            # the TOC title and would otherwise become orphan body blocks).
            # We match block text against the TOC title.
            title_words = set(toc_title.lower().split())
            while i < len(flat):
                candidate = flat[i]
                # Don't skip images
                if candidate.get("kind") == "_image":
                    break
                t = candidate["text"]
                # A block is a title fragment if it's short, single-line,
                # and its words overlap significantly with the TOC title
                if len(t) < 60 and candidate["nlines"] <= 1:
                    block_words = set(t.lower().split())
                    overlap = len(block_words & title_words)
                    if overlap >= max(1, len(block_words) * 0.4):
                        i += 1
                        continue
                break
            continue

        # Regular block
        tagged.append({
            "text": block["text"],
            "kind": "_raw",
            "chapter": current_chapter,
            "page": block["page"],
            "_merge": True,
        })
        i += 1

    # --- Pass 4: merge continuation blocks ---
    # If a raw block doesn't end with sentence-ending punctuation,
    # merge the next raw block into it. Works both within and across pages.
    merged: list[dict] = []
    for item in tagged:
        if item["kind"] in ("chapter_title", "image"):
            merged.append(item)
            continue

        if (
            merged
            and merged[-1].get("_merge")
            and not _ends_sentence(merged[-1]["text"])
        ):
            merged[-1]["text"] += " " + item["text"]
            merged[-1]["page"] = item["page"]
        else:
            merged.append(item)

    # --- Pass 5: classify merged blocks ---
    result: list[dict] = []
    for item in merged:
        if item["kind"] in ("chapter_title", "image"):
            result.append(item)
            continue

        text = item["text"]
        page = item["page"]

        # Caption: short text on a photo page
        if page in photo_pages and len(text) < 200:
            kind = "caption"
        # Caption: very short standalone text (pull quotes, epigraphs)
        elif len(text) < 60:
            kind = "caption"
        else:
            kind = "body"

        result.append({
            "text": text,
            "kind": kind,
            "chapter": item["chapter"],
            "page": page,
        })

    doc.close()
    return result


# ---------------------------------------------------------------------------
# Translation via LLM
# ---------------------------------------------------------------------------


def translate_batch(
    client: OpenAI,
    model: str,
    paragraphs: list[str],
    temperature: float = 0.3,
) -> list[str]:
    """
    Translate a batch of paragraphs in a single LLM call.
    Returns a list of translated strings (same order).
    """
    if len(paragraphs) == 1:
        user_msg = paragraphs[0]
    else:
        # Number paragraphs so the model returns them in order
        numbered = "\n\n".join(
            f"[{i+1}] {p}" for i, p in enumerate(paragraphs)
        )
        user_msg = (
            f"Translate each numbered paragraph below. "
            f"Keep the [N] markers in your output so I can parse them.\n\n"
            f"{numbered}"
        )

    # Estimate input tokens (~4 chars per token) and set max_tokens
    # to leave room for the translation (which is ~1.3x the input length).
    input_chars = len(SYSTEM_PROMPT) + len(user_msg)
    estimated_input_tokens = input_chars // 3  # conservative estimate
    max_output = max(1024, min(8192, 16384 - estimated_input_tokens))

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temperature,
                max_tokens=max_output,
            )
            content = resp.choices[0].message.content.strip()
            break
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"\n[ERROR] Translation failed after {MAX_RETRIES} retries: {e}")
                return ["[Translation error]"] * len(paragraphs)
            wait = 2 ** attempt
            print(f"\n[WARN] API error (attempt {attempt}/{MAX_RETRIES}): {e}. "
                  f"Retrying in {wait}s ...")
            time.sleep(wait)

    # Parse result
    if len(paragraphs) == 1:
        return [content]

    # Try to split by [N] markers
    results: list[str] = []
    pattern = re.compile(r"\[(\d+)\]\s*")
    parts = pattern.split(content)
    # parts looks like: ['', '1', 'text1', '2', 'text2', ...]
    # We take every odd-indexed-pair: (index_str, text)
    parsed: dict[int, str] = {}
    i = 1
    while i < len(parts) - 1:
        try:
            idx = int(parts[i])
            parsed[idx] = parts[i + 1].strip()
        except (ValueError, IndexError):
            pass
        i += 2

    for j in range(1, len(paragraphs) + 1):
        results.append(parsed.get(j, "[Translation error]"))

    return results


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------


class TranslatedPDF(FPDF):
    """Custom FPDF subclass for the bilingual book."""

    def __init__(self, font_regular: Path, font_bold: Path):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

        # Register Unicode fonts
        self.add_font("DejaVu", "", str(font_regular), uni=True)
        self.add_font("DejaVu", "B", str(font_bold), uni=True)

    def chapter_title(self, chapter_num: int | None, title_en: str, title_ru: str):
        """Add a bilingual chapter heading with chapter number."""
        # Start chapter on a new page
        self.add_page()

        # Chapter number
        if chapter_num is not None:
            self.set_font("DejaVu", "B", 28)
            self.set_text_color(180, 180, 180)
            self.cell(0, 16, f"{chapter_num:02d}", align="L", new_x="LMARGIN", new_y="NEXT")
            self.ln(4)

        # English title
        self.set_font("DejaVu", "B", 16)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 9, title_en, align="L")
        self.ln(2)

        # Russian title
        self.set_font("DejaVu", "B", 16)
        self.set_text_color(0, 80, 160)
        self.multi_cell(0, 9, title_ru, align="L")
        self.ln(8)

    def caption_pair(self, original: str, translated: str):
        """Add a caption/quote pair (italic, indented)."""
        self.set_font("DejaVu", "", 9)
        self.set_text_color(100, 100, 100)
        self.set_x(self.l_margin + 10)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 20, 5, original, align="L")
        self.ln(1)

        self.set_font("DejaVu", "", 10)
        self.set_text_color(0, 60, 120)
        self.set_x(self.l_margin + 10)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 20, 5.5, translated, align="L")
        self.ln(4)

    def paragraph_pair(self, original: str, translated: str):
        """Add an original paragraph (gray) + translated paragraph (black)."""
        # Original — smaller, gray
        self.set_font("DejaVu", "", 9)
        self.set_text_color(120, 120, 120)
        self.multi_cell(0, 5, original, align="L")
        self.ln(1)

        # Translation — normal size, dark
        self.set_font("DejaVu", "", 11)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 6, translated, align="L")
        self.ln(5)


def build_pdf(
    paragraphs: list[dict],
    translations: list[str],
    output_path: str,
    font_regular: Path,
    font_bold: Path,
):
    """Assemble the final bilingual PDF."""
    pdf = TranslatedPDF(font_regular, font_bold)
    pdf.add_page()

    # Title page
    pdf.set_font("DejaVu", "B", 22)
    pdf.set_text_color(30, 30, 30)
    pdf.ln(40)
    pdf.multi_cell(0, 12, "My Story", align="C")
    pdf.ln(4)
    pdf.set_font("DejaVu", "", 14)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 8, "Mohammed bin Rashid Al Maktoum", align="C")
    pdf.ln(8)
    pdf.set_font("DejaVu", "", 12)
    pdf.set_text_color(0, 80, 160)
    pdf.multi_cell(0, 8, "English / Русский — двойной перевод", align="C")

    pdf.add_page()

    usable_w = pdf.w - pdf.l_margin - pdf.r_margin

    for para, translation in zip(paragraphs, translations):
        kind = para.get("kind", "body")

        # --- Image entry ---
        if kind == "image":
            img_path = para.get("image_path", "")
            if img_path and os.path.exists(img_path):
                # Scale image to fit page width, max 60% of page height
                iw = para.get("img_width", 200)
                ih = para.get("img_height", 150)
                max_h = (pdf.h - pdf.t_margin - pdf.b_margin) * 0.6
                scale = min(usable_w / iw, max_h / ih, 1.0)
                draw_w = iw * scale
                draw_h = ih * scale

                # New page if not enough space
                if pdf.get_y() + draw_h + 10 > pdf.h - pdf.b_margin:
                    pdf.add_page()

                # Center image
                x = pdf.l_margin + (usable_w - draw_w) / 2
                pdf.image(img_path, x=x, y=pdf.get_y(), w=draw_w, h=draw_h)
                pdf.set_y(pdf.get_y() + draw_h + 4)
            continue

        # Check if we need a new page (leave at least 40mm)
        if pdf.get_y() > pdf.h - 40:
            pdf.add_page()

        if kind == "chapter_title":
            pdf.chapter_title(para.get("chapter"), para["text"], translation)
        elif kind == "caption":
            pdf.caption_pair(para["text"], translation)
        else:
            pdf.paragraph_pair(para["text"], translation)

    pdf.output(output_path)
    print(f"\nOutput saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Translate a PDF book (EN->RU) with interleaved bilingual output."
    )
    p.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the source PDF file.",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Path for the output PDF (default: <input>_translated.pdf).",
    )
    p.add_argument(
        "--api-url",
        default="http://localhost:8000/v1",
        help="Base URL of the vLLM OpenAI-compatible API (default: http://localhost:8000/v1).",
    )
    p.add_argument(
        "--model", "-m",
        default="Qwen/Qwen2.5-72B-Instruct",
        help="Model name on the vLLM server.",
    )
    p.add_argument(
        "--api-key",
        default="not-needed",
        help="API key (vLLM usually doesn't require one, but can be set).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of paragraphs per translation request (default: {BATCH_SIZE}).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for translation (default: 0.3).",
    )
    p.add_argument(
        "--resume",
        default=None,
        help="Path to a JSON checkpoint to resume from (skips already-translated paragraphs).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpointing (resume support)
# ---------------------------------------------------------------------------

CHECKPOINT_FILE = "translation_checkpoint.json"


def save_checkpoint(paragraphs: list[dict], translations: list[str | None], path: str):
    """Save current progress so we can resume later."""
    data = {
        "paragraphs": paragraphs,
        "translations": translations,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_checkpoint(path: str) -> tuple[list[dict], list[str | None]] | None:
    """Load checkpoint if it exists."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["paragraphs"], data["translations"]


def translate_all_with_checkpoint(
    client: OpenAI,
    model: str,
    paragraphs: list[dict],
    batch_size: int,
    temperature: float,
    checkpoint_path: str,
) -> list[str]:
    """
    Translate all paragraphs with checkpoint/resume support.
    Saves progress after each batch so work isn't lost if interrupted.
    """
    # Try loading existing checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is not None:
        saved_paras, saved_translations = checkpoint
        if len(saved_paras) == len(paragraphs):
            translations = saved_translations
            done = sum(1 for t in translations if t is not None)
            print(f"Resumed from checkpoint: {done}/{len(paragraphs)} paragraphs already translated.")
        else:
            print("Checkpoint paragraph count mismatch — starting fresh.")
            translations: list[str | None] = [None] * len(paragraphs)
    else:
        translations: list[str | None] = [None] * len(paragraphs)

    # Mark short paragraphs and images as "translated" (keep as-is / skip)
    for i, p in enumerate(paragraphs):
        if translations[i] is not None:
            continue
        if p.get("kind") == "image":
            translations[i] = ""
        elif len(p["text"]) < MIN_PARAGRAPH_LEN:
            translations[i] = p["text"]

    # Collect indices that still need translation
    remaining = [
        (i, paragraphs[i])
        for i in range(len(paragraphs))
        if translations[i] is None
    ]

    if not remaining:
        print("All paragraphs already translated!")
        return translations  # type: ignore

    # Batch and translate
    batches: list[list[tuple[int, dict]]] = []
    for start in range(0, len(remaining), batch_size):
        batches.append(remaining[start : start + batch_size])

    pbar = tqdm(total=len(remaining), desc="Translating", unit="para")

    for batch in batches:
        indices = [idx for idx, _ in batch]
        texts = [p["text"] for _, p in batch]

        translated = translate_batch(client, model, texts, temperature)

        for idx, tr in zip(indices, translated):
            translations[idx] = tr
            pbar.update(1)

        # Save checkpoint after each batch
        save_checkpoint(paragraphs, translations, checkpoint_path)

    pbar.close()
    return translations  # type: ignore


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # Output path
    if args.output is None:
        stem = Path(args.input).stem
        args.output = str(Path(args.input).parent / f"{stem}_translated.pdf")

    # Checkpoint path
    checkpoint_path = args.resume or str(
        Path(args.input).parent / CHECKPOINT_FILE
    )

    # 1. Ensure fonts
    print("=== Step 1/4: Checking fonts ===")
    font_regular, font_bold = ensure_fonts()

    # 2. Extract text
    print(f"\n=== Step 2/4: Extracting text from {args.input} ===")
    paragraphs = extract_paragraphs(args.input)
    kinds = {}
    for p in paragraphs:
        k = p.get("kind", "body")
        kinds[k] = kinds.get(k, 0) + 1
    stats = ", ".join(f"{v} {k}" for k, v in sorted(kinds.items()))
    print(f"  Found {len(paragraphs)} text blocks ({stats})")

    if not paragraphs:
        print("ERROR: No text extracted from PDF. Is this a scanned/image PDF?")
        sys.exit(1)

    # 3. Translate
    print(f"\n=== Step 3/4: Translating via {args.model} ===")
    print(f"  API: {args.api_url}")
    print(f"  Batch size: {args.batch_size}")

    client = OpenAI(
        base_url=args.api_url,
        api_key=args.api_key,
    )

    translations = translate_all_with_checkpoint(
        client=client,
        model=args.model,
        paragraphs=paragraphs,
        batch_size=args.batch_size,
        temperature=args.temperature,
        checkpoint_path=checkpoint_path,
    )

    # 4. Build PDF
    print(f"\n=== Step 4/4: Building bilingual PDF ===")
    build_pdf(paragraphs, translations, args.output, font_regular, font_bold)

    # Clean up checkpoint on success
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint removed (translation complete).")

    print("\nDone!")


if __name__ == "__main__":
    main()
