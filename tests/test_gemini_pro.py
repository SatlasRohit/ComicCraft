import json

import pytest
from pydantic import ValidationError

import app.gemini_pro as gemini_pro
from app.gemini_pro import StoryResponse


class FakeResponse:
    def __init__(self, parsed=None, text=""):
        self.parsed = parsed
        self.text = text


class FakeModels:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class FakeClient:
    def __init__(self, response):
        self.models = FakeModels(response)


def make_outline():
    return [
        {
            "panel_number": i,
            "title": f"Panel {i}",
            "scene_description": f"Scene {i}",
            "image_prompt": f"image prompt {i}",
        }
        for i in range(1, 6)
    ]


def make_story_panels():
    return [
        {
            "panel_number": i,
            "title": f"Panel {i}",
            "scene_description": f"Scene {i}",
            "narration": f"Narration {i}",
            "dialogue": f"Dialogue {i}",
            "image_prompt": f"image prompt {i}",
        }
        for i in range(1, 6)
    ]


@pytest.fixture
def patch_client(monkeypatch):
    def _patch(response):
        client = FakeClient(response)
        monkeypatch.setattr(gemini_pro, "_build_client", lambda: client)
        return client

    return _patch


def test_generate_story_returns_full_panels(patch_client):
    parsed = StoryResponse.model_validate({"panels": make_story_panels()})
    client = patch_client(FakeResponse(parsed=parsed))

    result = gemini_pro.generate_story(make_outline())

    assert len(result) == 5
    assert set(result[0]) == {
        "panel_number",
        "title",
        "scene_description",
        "narration",
        "dialogue",
        "image_prompt",
    }
    assert result[0]["narration"] == "Narration 1"
    assert result[0]["dialogue"] == "Dialogue 1"
    call = client.models.calls[0]
    assert call["model"] == gemini_pro.MODEL
    assert call["config"].response_mime_type == "application/json"


def test_generate_story_parses_json_text_fallback(patch_client):
    payload = json.dumps({"panels": make_story_panels()})
    patch_client(FakeResponse(parsed=None, text=payload))

    result = gemini_pro.generate_story(make_outline())

    assert len(result) == 5
    assert result[4]["panel_number"] == 5


def test_generate_story_rejects_missing_dialogue(patch_client):
    panels = make_story_panels()
    del panels[0]["dialogue"]
    payload = json.dumps({"panels": panels})
    patch_client(FakeResponse(parsed=None, text=payload))

    with pytest.raises(ValidationError):
        gemini_pro.generate_story(make_outline())


def test_generate_story_rejects_wrong_panel_count(patch_client):
    payload = json.dumps({"panels": make_story_panels()[:4]})
    patch_client(FakeResponse(parsed=None, text=payload))

    with pytest.raises(ValidationError):
        gemini_pro.generate_story(make_outline())


def test_build_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        gemini_pro._build_client()
