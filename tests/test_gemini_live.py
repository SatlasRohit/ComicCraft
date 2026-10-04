import os

import pytest

from app.gemini_flash import generate_outline
from app.gemini_pro import generate_story

pytestmark = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set; skipping live Gemini API test",
)


def test_live_outline_and_story():
    try:
        outline = generate_outline(
            "A brave fox exploring an enchanted forest",
            "Rusty",
            "forest",
            "dramatic",
            "anime",
        )
    except Exception as exc:
        if (
            getattr(exc, "status_code", None) in (401, 403, 404, 429, 500, 503)
            or "Connect" in type(exc).__name__
            or "Timeout" in type(exc).__name__
        ):
            pytest.skip(f"live Gemini call skipped ({type(exc).__name__}): {exc}")
        raise

    assert len(outline) == 5

    try:
        story = generate_story(outline)
    except Exception as exc:
        if (
            getattr(exc, "status_code", None) in (401, 403, 404, 429, 500, 503)
            or "Connect" in type(exc).__name__
            or "Timeout" in type(exc).__name__
        ):
            pytest.skip(f"live Gemini call skipped ({type(exc).__name__}): {exc}")
        raise

    assert len(story) == 5
    for panel in story:
        assert panel["panel_number"]
        assert panel["narration"]
        assert panel["dialogue"]
