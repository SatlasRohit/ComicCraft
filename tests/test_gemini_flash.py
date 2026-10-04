import json

import pytest
from pydantic import ValidationError

import app.gemini_flash as gemini_flash
from app.gemini_flash import OutlineResponse


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


def make_panels(count=5):
    return [
        {
            "panel_number": i,
            "title": f"Panel {i}",
            "scene_description": f"Scene {i}",
            "image_prompt": f"image prompt {i}",
        }
        for i in range(1, count + 1)
    ]


@pytest.fixture
def patch_client(monkeypatch):
    def _patch(response):
        client = FakeClient(response)
        monkeypatch.setattr(gemini_flash, "_build_client", lambda: client)
        return client

    return _patch


def test_generate_outline_returns_five_panel_dicts(patch_client):
    panels = make_panels()
    parsed = OutlineResponse.model_validate({"panels": panels})
    client = patch_client(FakeResponse(parsed=parsed))

    result = gemini_flash.generate_outline("A fox adventure", "Rusty", "forest", "dramatic", "anime")

    assert isinstance(result, list)
    assert len(result) == 5
    assert [p["panel_number"] for p in result] == [1, 2, 3, 4, 5]
    assert set(result[0]) == {"panel_number", "title", "scene_description", "image_prompt"}
    call = client.models.calls[0]
    assert call["model"] == gemini_flash.MODEL
    assert call["config"].response_mime_type == "application/json"


def test_generate_outline_parses_json_text_fallback(patch_client):
    payload = json.dumps({"panels": make_panels()})
    patch_client(FakeResponse(parsed=None, text=payload))

    result = gemini_flash.generate_outline("A fox adventure", "Rusty", "forest", "dramatic", "anime")

    assert len(result) == 5
    assert result[4]["panel_number"] == 5


def test_generate_outline_rejects_wrong_panel_count(patch_client):
    payload = json.dumps({"panels": make_panels(count=4)})
    patch_client(FakeResponse(parsed=None, text=payload))

    with pytest.raises(ValidationError):
        gemini_flash.generate_outline("A fox adventure", "Rusty", "forest", "dramatic", "anime")


def test_generate_outline_rejects_empty_response(patch_client):
    patch_client(FakeResponse(parsed=None, text=""))

    with pytest.raises(ValueError):
        gemini_flash.generate_outline("A fox adventure", "Rusty", "forest", "dramatic", "anime")


def test_build_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        gemini_flash._build_client()
