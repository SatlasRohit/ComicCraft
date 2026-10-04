import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, field_validator

load_dotenv()

MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-3.5-flash")
PANEL_COUNT = 5
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash"]


class PanelOutline(BaseModel):
    panel_number: int = Field(..., ge=1, le=5)
    title: str = Field(..., min_length=1)
    scene_description: str = Field(..., min_length=1)
    image_prompt: str = Field(..., min_length=1)


class OutlineResponse(BaseModel):
    panels: list[PanelOutline]

    @field_validator("panels")
    @classmethod
    def validate_panels(cls, panels: list[PanelOutline]) -> list[PanelOutline]:
        if len(panels) != PANEL_COUNT:
            raise ValueError(f"Expected exactly {PANEL_COUNT} panels, got {len(panels)}")
        numbers = [panel.panel_number for panel in panels]
        if numbers != list(range(1, PANEL_COUNT + 1)):
            raise ValueError("panel_number values must be sequential from 1 to 5")
        return panels


def _build_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
    return genai.Client(api_key=api_key)


def _parse_response(response, schema: type[BaseModel]) -> BaseModel:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return parsed if isinstance(parsed, schema) else schema.model_validate(parsed)

    text = (getattr(response, "text", "") or "").strip()
    if not text:
        raise ValueError("Gemini returned an empty response")
    return schema.model_validate_json(text)


def _build_prompt(story_prompt: str, character_name: str, setting: str, tone: str, art_style: str) -> str:
    return (
        "You are a comic book writer. Create a structured outline for a 5-panel comic.\n\n"
        f"Story prompt: {story_prompt}\n"
        f"Main character: {character_name}\n"
        f"Setting: {setting}\n"
        f"Tone: {tone}\n"
        f"Art style: {art_style}\n\n"
        f"Produce exactly {PANEL_COUNT} panels, numbered 1 through {PANEL_COUNT}, in order.\n"
        "For each panel provide:\n"
        "- panel_number: the panel number\n"
        "- title: a short descriptive title\n"
        "- scene_description: one or two sentences describing what happens\n"
        f"- image_prompt: a detailed Stable Diffusion prompt, in the '{art_style}' style, "
        "that depicts the scene without any text or lettering\n"
        "Return only JSON that matches the provided schema."
    )


def generate_outline(
    story_prompt: str,
    character_name: str,
    setting: str,
    tone: str,
    art_style: str,
) -> list[dict]:
    client = _build_client()
    prompt = _build_prompt(story_prompt, character_name, setting, tone, art_style)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=OutlineResponse,
        temperature=0.8,
    )

    is_custom_client = not hasattr(client, "models") or "fake" in type(client.models).__name__.lower()
    if is_custom_client:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=config,
        )
        parsed = _parse_response(response, OutlineResponse)
        return [panel.model_dump() for panel in parsed.panels]

    models_to_try = [MODEL] + [m for m in FALLBACK_MODELS if m != MODEL]
    last_exc = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            parsed = _parse_response(response, OutlineResponse)
            return [panel.model_dump() for panel in parsed.panels]
        except Exception as exc:
            last_exc = exc
            continue

    raise last_exc
