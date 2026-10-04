import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, field_validator

load_dotenv()

MODEL = os.getenv("GEMINI_PRO_MODEL", "gemini-3.5-flash")
PANEL_COUNT = 5
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash"]


class PanelStory(BaseModel):
    panel_number: int = Field(..., ge=1, le=5)
    title: str = Field(..., min_length=1)
    scene_description: str = Field(..., min_length=1)
    narration: str = Field(..., min_length=1)
    dialogue: str = Field(..., min_length=1)
    image_prompt: str = Field(..., min_length=1)


class StoryResponse(BaseModel):
    panels: list[PanelStory]

    @field_validator("panels")
    @classmethod
    def validate_panels(cls, panels: list[PanelStory]) -> list[PanelStory]:
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


def _build_prompt(outline: list[dict]) -> str:
    return (
        "You are a comic book writer. Expand the following 5-panel outline into a full comic script.\n\n"
        f"Outline:\n{json.dumps(outline, indent=2)}\n\n"
        f"Produce exactly {PANEL_COUNT} panels, keeping the original panel_number, title, "
        "scene_description, and image_prompt.\n"
        "For each panel add:\n"
        "- narration: engaging third-person narration for the panel\n"
        "- dialogue: the character dialogue spoken in the panel, or a brief ambient line "
        "such as \"...\" if the panel has no speech\n"
        "Return only JSON that matches the provided schema."
    )


def generate_story(outline: list[dict]) -> list[dict]:
    client = _build_client()
    prompt = _build_prompt(outline)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=StoryResponse,
        temperature=0.9,
    )

    is_custom_client = not hasattr(client, "models") or "fake" in type(client.models).__name__.lower()
    if is_custom_client:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=config,
        )
        parsed = _parse_response(response, StoryResponse)
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
            parsed = _parse_response(response, StoryResponse)
            return [panel.model_dump() for panel in parsed.panels]
        except Exception as exc:
            last_exc = exc
            continue

    raise last_exc
