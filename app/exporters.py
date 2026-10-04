import os
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fpdf import FPDF
from PIL import Image

load_dotenv()

DEFAULT_EXPORT_DIR = "static/exports"
EXPORT_FIELDS = (
    "panel_number",
    "title",
    "scene_description",
    "narration",
    "dialogue",
    "image_path",
)
MAX_IMAGE_HEIGHT_MM = 110

_PUNCT_REPLACEMENTS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
    "\u00a0": " ",
}


def export_dir() -> Path:
    directory = Path(os.getenv("PDF_OUTPUT_DIR", DEFAULT_EXPORT_DIR))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _latin1(text) -> str:
    text = "" if text is None else str(text)
    for source, replacement in _PUNCT_REPLACEMENTS.items():
        text = text.replace(source, replacement)
    try:
        text.encode("latin-1")
    except UnicodeEncodeError:
        text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text


def _validate_layout(comic_layout) -> list[dict]:
    if not isinstance(comic_layout, list) or not comic_layout:
        raise ValueError("comic_layout must be a non-empty list of panel dicts")

    panels = []
    for index, panel in enumerate(comic_layout, start=1):
        if not isinstance(panel, dict):
            raise ValueError(f"panel {index} must be a dict, got {type(panel).__name__}")
        missing = [field for field in EXPORT_FIELDS if field not in panel]
        if missing:
            raise ValueError(
                f"panel {index} is missing fields required for PDF export: {', '.join(missing)}"
            )
        panels.append(panel)
    return panels


def _note_placeholder(pdf: FPDF, message: str) -> None:
    pdf.set_font("helvetica", "I", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 6, _latin1(message))
    pdf.set_text_color(0, 0, 0)


def _place_image(pdf: FPDF, image_path) -> None:
    path = Path(image_path)
    if not path.is_file():
        _note_placeholder(pdf, f"[Panel image not found: {path}]")
        return

    try:
        with Image.open(path) as image:
            width_px, height_px = image.size

        ratio = height_px / width_px
        width = min(pdf.epw, MAX_IMAGE_HEIGHT_MM / ratio)
        height = width * ratio
        x = pdf.l_margin + (pdf.epw - width) / 2
        y = pdf.get_y()

        pdf.image(str(path), x=x, y=y, w=width, h=height)
        pdf.set_y(y + height)
    except Exception as exc:
        _note_placeholder(pdf, f"[Panel image could not be embedded: {exc}]")


def _add_panel_page(pdf: FPDF, panel: dict, font_family: str = "helvetica") -> None:
    pdf.add_page()

    pdf.set_font(font_family, "B", 16)
    title = panel["title"] or f"Panel {panel['panel_number']}"
    if font_family == "helvetica":
        title = _latin1(title)
        header_text = _latin1(f"Panel {panel['panel_number']}: {title}")
    else:
        header_text = f"Panel {panel['panel_number']}: {title}"
    pdf.multi_cell(0, 8, header_text)
    pdf.ln(2)

    _place_image(pdf, panel["image_path"])
    pdf.ln(4)

    pdf.set_font(font_family, "I", 11)
    pdf.set_text_color(100, 116, 139)
    scene = _latin1(panel["scene_description"]) if font_family == "helvetica" else panel["scene_description"]
    pdf.multi_cell(0, 6, scene)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    pdf.set_font(font_family, "", 12)
    narration = _latin1(panel["narration"]) if font_family == "helvetica" else panel["narration"]
    pdf.multi_cell(0, 6, narration)
    pdf.ln(3)

    pdf.set_font(font_family, "B", 12)
    dialogue = _latin1(f"Dialogue: {panel['dialogue']}") if font_family == "helvetica" else f"Dialogue: {panel['dialogue']}"
    pdf.multi_cell(0, 6, dialogue)


def save_pdf(comic_layout: list[dict]) -> str:
    panels = _validate_layout(comic_layout)
    directory = export_dir()
    filename = (
        f"comic_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.pdf"
    )
    path = directory / filename

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_title("ComicCraft Comic")

    font_path = Path("static/fonts/DejaVuSans.ttf")
    font_bold_path = Path("static/fonts/DejaVuSans-Bold.ttf")
    font_oblique_path = Path("static/fonts/DejaVuSans-Oblique.ttf")

    if font_path.is_file():
        try:
            pdf.add_font("DejaVu", "", str(font_path))
            if font_bold_path.is_file():
                pdf.add_font("DejaVu", "B", str(font_bold_path))
            if font_oblique_path.is_file():
                pdf.add_font("DejaVu", "I", str(font_oblique_path))
            font_family = "DejaVu"
        except Exception:
            font_family = "helvetica"
    else:
        font_family = "helvetica"

    for panel in panels:
        _add_panel_page(pdf, panel, font_family=font_family)

    try:
        pdf.output(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to write PDF to {path}: {exc}") from exc

    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"PDF was not written correctly: {path}")

    return str(path)

