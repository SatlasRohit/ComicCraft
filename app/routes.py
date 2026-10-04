import logging
import uuid
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.exporters import save_pdf
from app.gemini_flash import generate_outline
from app.gemini_pro import generate_story
from app.image_generator import generate_image
from app.layout_builder import build_comic_layout

logger = logging.getLogger("comiccraft.routes")

router = APIRouter()
templates = Jinja2Templates(directory="templates")

_COMIC_STORE: dict[str, list[dict]] = {}
_MAX_STORED_COMICS = 50

INPUT_LIMITS = {
    "story_prompt": 2000,
    "character_name": 100,
    "setting": 100,
    "tone": 100,
    "art_style": 100,
}


class PromptRequest(BaseModel):
    story_prompt: str | None = Field(default=None, max_length=INPUT_LIMITS["story_prompt"])
    prompt: str | None = Field(default=None, max_length=INPUT_LIMITS["story_prompt"])
    character_name: str = Field(..., min_length=1, max_length=INPUT_LIMITS["character_name"])
    setting: str = Field(..., min_length=1, max_length=INPUT_LIMITS["setting"])
    tone: str | None = Field(default=None, max_length=INPUT_LIMITS["tone"])
    story_tone: str | None = Field(default=None, max_length=INPUT_LIMITS["tone"])
    art_style: str | None = Field(default=None, max_length=INPUT_LIMITS["art_style"])
    style: str | None = Field(default=None, max_length=INPUT_LIMITS["art_style"])

    def get_story_prompt(self) -> str:
        return self.story_prompt or self.prompt or ""

    def get_tone(self) -> str:
        return self.tone or self.story_tone or ""

    def get_art_style(self) -> str:
        return self.art_style or self.style or ""



def register_comic(layout: list[dict]) -> str:
    comic_id = uuid.uuid4().hex
    _COMIC_STORE[comic_id] = layout
    while len(_COMIC_STORE) > _MAX_STORED_COMICS:
        oldest = next(iter(_COMIC_STORE))
        del _COMIC_STORE[oldest]
    return comic_id


def get_comic(comic_id: str) -> list[dict] | None:
    return _COMIC_STORE.get(comic_id)


def _static_url(path: str) -> str:
    url = str(path).replace("\\", "/")
    static_index = url.find("static/")
    if static_index >= 0:
        return "/" + url[static_index:]
    return url if url.startswith("/") else "/" + url


def _validate_inputs(story_prompt, character_name, setting, tone, art_style) -> None:
    for label, value in (
        ("story_prompt", story_prompt),
        ("character_name", character_name),
        ("setting", setting),
        ("tone", tone),
        ("art_style", art_style),
    ):
        if value is None or not str(value).strip():
            raise HTTPException(status_code=422, detail=f"{label} is required.")
        if len(value) > INPUT_LIMITS[label]:
            raise HTTPException(
                status_code=422,
                detail=f"{label} is too long (max {INPUT_LIMITS[label]} characters).",
            )


def _story_error_detail(stage_message: str, exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return "The Gemini API key was rejected. Ask the administrator to check GEMINI_API_KEY."
    if status == 404:
        return (
            "The configured Gemini model is no longer available. "
            "Ask the administrator to update GEMINI_FLASH_MODEL/GEMINI_PRO_MODEL."
        )
    if status == 429:
        return (
            "Gemini quota was exceeded for this request. Please try again later, "
            "or ask the administrator to check the API plan or model settings."
        )
    if status in (500, 503):
        return (
            "Gemini is temporarily unavailable due to high demand. Please try again in a moment."
        )
    return stage_message


def _generate_comic(story_prompt, character_name, setting, tone, art_style) -> tuple[str, list[dict]]:
    _validate_inputs(story_prompt, character_name, setting, tone, art_style)

    try:
        outline = generate_outline(story_prompt, character_name, setting, tone, art_style)
    except RuntimeError as exc:
        logger.error("Comic generation is not configured: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="The comic service is not configured. Contact the administrator.",
        )
    except Exception as exc:
        logger.exception("Gemini outline generation failed")
        raise HTTPException(
            status_code=502,
            detail=_story_error_detail(
                "Failed to generate the story outline. Please try again.", exc
            ),
        )

    try:
        story = generate_story(outline)
    except Exception as exc:
        logger.exception("Gemini story generation failed")
        raise HTTPException(
            status_code=502,
            detail=_story_error_detail(
                "Failed to write the comic story. Please try again.", exc
            ),
        )

    try:
        image_paths = [
            generate_image(panel["image_prompt"], panel["panel_number"])
            for panel in story
        ]
    except Exception:
        logger.exception("Panel image generation failed")
        raise HTTPException(
            status_code=502,
            detail="Failed to illustrate the comic panels. Please try again.",
        )

    try:
        layout = build_comic_layout(story, image_paths)
    except ValueError:
        logger.exception("Comic layout assembly failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to assemble the comic layout.",
        )

    return register_comic(layout), layout


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"title": "ComicCraft"})


@router.post("/generate", response_class=HTMLResponse)
def generate_comic_page(
    request: Request,
    story_prompt: str | None = Form(None),
    prompt: str | None = Form(None),
    character_name: str = Form(...),
    setting: str = Form(...),
    tone: str | None = Form(None),
    story_tone: str | None = Form(None),
    art_style: str | None = Form(None),
    style: str | None = Form(None),
):
    effective_prompt = (story_prompt if story_prompt is not None and str(story_prompt).strip() else prompt) or ""
    effective_tone = (tone if tone is not None and str(tone).strip() else story_tone) or ""
    effective_style = (art_style if art_style is not None and str(art_style).strip() else style) or ""

    try:
        comic_id, layout = _generate_comic(
            effective_prompt, character_name, setting, effective_tone, effective_style
        )
        return templates.TemplateResponse(
            request,
            "comic_preview.html",
            {"title": "ComicCraft", "layout": layout, "comic_id": comic_id},
        )
    except HTTPException as exc:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "title": "ComicCraft",
                "error_message": exc.detail,
                "story_prompt": effective_prompt,
                "character_name": character_name,
                "setting": setting,
                "tone": effective_tone,
                "art_style": effective_style,
            },
            status_code=exc.status_code,
        )


@router.post("/generate-comic/json")
def generate_comic_json(payload: PromptRequest):
    comic_id, layout = _generate_comic(
        payload.get_story_prompt(),
        payload.character_name,
        payload.setting,
        payload.get_tone(),
        payload.get_art_style(),
    )

    try:
        pdf_path = save_pdf(layout)
    except Exception:
        logger.exception("PDF export failed for comic %s", comic_id)
        raise HTTPException(
            status_code=500,
            detail="The comic was generated, but the PDF export failed.",
        )

    return JSONResponse(
        {
            "comic_id": comic_id,
            "panels": layout,
            "pdf_path": pdf_path,
            "pdf_url": _static_url(pdf_path),
        }
    )


@router.get("/test-image")
def test_image(prompt: str = Query(..., min_length=1, max_length=INPUT_LIMITS["story_prompt"])):
    try:
        image_path = generate_image(prompt, panel_number=1)
    except Exception:
        logger.exception("Test image generation failed")
        raise HTTPException(
            status_code=502,
            detail="Image generation failed. Please try again.",
        )

    return JSONResponse(
        {
            "message": "Image generated successfully",
            "path": image_path,
            "image_path": image_path,
            "image_url": _static_url(image_path),
        }
    )


@router.post("/export")
def export_comic(comic_id: str = Form(...)):
    layout = get_comic(comic_id)
    if layout is None:
        raise HTTPException(
            status_code=404, detail="Comic not found. Generate a comic first."
        )

    try:
        pdf_path = save_pdf(layout)
    except Exception:
        logger.exception("PDF export failed for comic %s", comic_id)
        raise HTTPException(
            status_code=500,
            detail="PDF export failed. Please try again.",
        )

    filename = Path(pdf_path).name
    return RedirectResponse(
        f"/export-success?pdf={quote(filename)}", status_code=303
    )


@router.get("/export-success", response_class=HTMLResponse)
async def export_success(request: Request, pdf: str | None = None):
    pdf_name = None
    pdf_url = None
    if pdf:
        candidate = Path(pdf).name
        if candidate.lower().endswith(".pdf"):
            pdf_name = candidate
            pdf_url = f"/static/exports/{candidate}"

    return templates.TemplateResponse(
        request,
        "export_success.html",
        {"title": "ComicCraft", "pdf_name": pdf_name, "pdf_url": pdf_url},
    )
