from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.routes as routes
from app.main import app

FORM = {
    "story_prompt": "A brave fox exploring an enchanted forest",
    "character_name": "Rusty",
    "setting": "forest",
    "tone": "dramatic",
    "art_style": "anime",
}


@pytest.fixture(autouse=True)
def exports_tmp(tmp_path, monkeypatch):
    directory = tmp_path / "static" / "exports"
    monkeypatch.setenv("PDF_OUTPUT_DIR", str(directory))
    return directory


@pytest.fixture
def client():
    return TestClient(app)


def make_outline():
    return [
        {
            "panel_number": i,
            "title": f"Outline Title {i}",
            "scene_description": f"Scene {i}",
            "image_prompt": f"image prompt {i}",
        }
        for i in range(1, 6)
    ]


def make_story():
    return [
        {
            **panel,
            "narration": f"Narration {panel['panel_number']}",
            "dialogue": f"Dialogue {panel['panel_number']}",
        }
        for panel in make_outline()
    ]


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    panels_dir = tmp_path / "static" / "panels"
    panels_dir.mkdir(parents=True)

    def fake_generate_image(image_prompt, panel_number):
        path = panels_dir / f"panel_{panel_number}.png"
        Image.new("RGB", (32, 24), (80, 120, 200)).save(path, format="PNG")
        return str(path)

    monkeypatch.setattr(routes, "generate_outline", lambda *a, **k: make_outline())
    monkeypatch.setattr(routes, "generate_story", lambda outline: make_story())
    monkeypatch.setattr(routes, "generate_image", fake_generate_image)
    return panels_dir


def test_index_form_uses_story_prompt_field(client):
    response = client.get("/")

    assert response.status_code == 200
    assert 'name="story_prompt"' in response.text
    assert 'href="/static/css/styles.css"' in response.text
    assert 'id="loading-overlay"' in response.text


def test_generate_renders_preview(client, pipeline):
    response = client.post("/generate", data=FORM)

    assert response.status_code == 200
    html = response.text
    assert "Outline Title 1" in html
    assert "Narration 3" in html
    assert "Dialogue 5" in html
    assert "/static/panels/panel_2.png" in html
    assert html.count('name="comic_id"') == 2


def test_generate_missing_field_returns_422(client, pipeline):
    data = dict(FORM)
    del data["tone"]

    response = client.post("/generate", data=data)

    assert response.status_code == 422


def test_generate_blank_prompt_returns_422(client, pipeline):
    response = client.post("/generate", data={**FORM, "story_prompt": "   "})

    assert response.status_code == 422


def test_generate_missing_env_returns_503_without_leaking_details(client, monkeypatch):
    def unconfigured(*args, **kwargs):
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")

    monkeypatch.setattr(routes, "generate_outline", unconfigured)

    response = client.post("/generate", data=FORM)

    assert response.status_code == 503
    assert "not configured" in response.text
    assert "GEMINI_API_KEY" not in response.text


def test_generate_gemini_failure_returns_502_without_internals(client, monkeypatch):
    class FakeGeminiAPIError(Exception):
        pass

    def failing(*args, **kwargs):
        raise FakeGeminiAPIError("upstream 429, project=secret-project, key=AIzaFAKE")

    monkeypatch.setattr(routes, "generate_outline", failing)

    response = client.post("/generate", data=FORM)

    assert response.status_code == 502
    assert "Failed to generate the story outline" in response.text
    assert "AIzaFAKE" not in response.text
    assert "secret-project" not in response.text
    assert "Traceback" not in response.text


def test_generate_story_failure_returns_502(client, pipeline, monkeypatch):
    def failing(outline):
        raise RuntimeError("story model overloaded")

    monkeypatch.setattr(routes, "generate_story", failing)

    response = client.post("/generate", data=FORM)

    assert response.status_code == 502
    assert "Failed to write the comic story" in response.text


def test_generate_image_failure_returns_502(client, pipeline, monkeypatch):
    def failing(image_prompt, panel_number):
        raise RuntimeError("cuda out of memory")

    monkeypatch.setattr(routes, "generate_image", failing)

    response = client.post("/generate", data=FORM)

    assert response.status_code == 502
    assert "Failed to illustrate the comic panels" in response.text


def test_generate_layout_failure_returns_500(client, pipeline, monkeypatch):
    def failing(story, image_paths):
        raise ValueError("bad layout")

    monkeypatch.setattr(routes, "build_comic_layout", failing)

    response = client.post("/generate", data=FORM)

    assert response.status_code == 500
    assert "Failed to assemble the comic layout" in response.text


def test_generate_comic_json_full_workflow(client, pipeline, exports_tmp):
    response = client.post("/generate-comic/json", json=dict(FORM))

    assert response.status_code == 200
    body = response.json()

    assert body["comic_id"]
    assert len(body["panels"]) == 5
    assert set(body["panels"][0]) == {
        "panel_number",
        "title",
        "scene_description",
        "narration",
        "dialogue",
        "image_prompt",
        "image_path",
        "image_url",
    }
    assert body["pdf_url"].startswith("/static/exports/comic_")
    assert Path(body["pdf_path"]).is_file()
    assert routes.get_comic(body["comic_id"]) == body["panels"]


def test_generate_comic_json_invalid_body_returns_422(client, pipeline):
    response = client.post("/generate-comic/json", json={"story_prompt": "only one field"})

    assert response.status_code == 422


def test_generate_comic_json_missing_env_returns_503(client, monkeypatch):
    def unconfigured(*args, **kwargs):
        raise RuntimeError("GEMINI_API_KEY is not set")

    monkeypatch.setattr(routes, "generate_outline", unconfigured)

    response = client.post("/generate-comic/json", json=dict(FORM))

    assert response.status_code == 503


def test_generate_comic_json_pdf_failure_returns_500(client, pipeline, monkeypatch):
    def failing(layout):
        raise RuntimeError("disk full")

    monkeypatch.setattr(routes, "save_pdf", failing)

    response = client.post("/generate-comic/json", json=dict(FORM))

    assert response.status_code == 500
    assert "PDF export failed" in response.text


def test_test_image_success(client, pipeline):
    response = client.get("/test-image", params={"prompt": "a brave fox"})

    assert response.status_code == 200
    body = response.json()
    assert body["image_url"] == "/static/panels/panel_1.png"
    assert Path(body["image_path"]).is_file()


def test_test_image_requires_prompt(client):
    response = client.get("/test-image")

    assert response.status_code == 422


def test_test_image_failure_returns_502(client, monkeypatch):
    def failing(image_prompt, panel_number):
        raise RuntimeError("model download failed")

    monkeypatch.setattr(routes, "generate_image", failing)

    response = client.get("/test-image", params={"prompt": "a brave fox"})

    assert response.status_code == 502
    assert "Image generation failed" in response.text


def test_end_to_end_generate_then_export(client, pipeline, exports_tmp):
    generated = client.post("/generate-comic/json", json=dict(FORM))
    comic_id = generated.json()["comic_id"]

    exported = client.post("/export", data={"comic_id": comic_id}, follow_redirects=True)

    assert exported.status_code == 200
    assert "/export-success" in str(exported.url)
    assert "Download PDF" in exported.text
    assert len(list(exports_tmp.glob("comic_*.pdf"))) == 2


def test_generate_rejects_oversized_prompt(client, pipeline):
    response = client.post("/generate", data={**FORM, "story_prompt": "x" * 2001})

    assert response.status_code == 422


def test_generate_comic_json_rejects_oversized_prompt(client, pipeline):
    response = client.post("/generate-comic/json", json={**FORM, "story_prompt": "x" * 2001})

    assert response.status_code == 422


def test_generate_comic_json_rejects_whitespace_only_fields(client, pipeline):
    response = client.post("/generate-comic/json", json={**FORM, "character_name": "   "})

    assert response.status_code == 422


def test_test_image_rejects_oversized_prompt(client):
    response = client.get("/test-image", params={"prompt": "x" * 2001})

    assert response.status_code == 422


class FakeAPIError(Exception):
    def __init__(self, status_code):
        super().__init__(f"fake upstream error {status_code}")
        self.status_code = status_code


def test_generate_quota_error_informs_user(client, monkeypatch):
    def quota_blocked(*args, **kwargs):
        raise FakeAPIError(429)

    monkeypatch.setattr(routes, "generate_outline", quota_blocked)

    response = client.post("/generate", data=FORM)

    assert response.status_code == 502
    assert "quota was exceeded" in response.text


def test_generate_retired_model_error_informs_user(client, monkeypatch):
    def model_gone(*args, **kwargs):
        raise FakeAPIError(404)

    monkeypatch.setattr(routes, "generate_outline", model_gone)

    response = client.post("/generate", data=FORM)

    assert response.status_code == 502
    assert "no longer available" in response.text
    assert "GEMINI_FLASH_MODEL" in response.text


def test_generate_key_rejected_error_informs_user(client, pipeline, monkeypatch):
    def key_rejected(*args, **kwargs):
        raise FakeAPIError(401)

    monkeypatch.setattr(routes, "generate_story", key_rejected)

    response = client.post("/generate", data=FORM)

    assert response.status_code == 502
    assert "API key was rejected" in response.text


def test_comic_store_evicts_oldest_beyond_cap(client):
    routes._COMIC_STORE.clear()
    first = routes.register_comic([{"panel_number": 1}])
    for _ in range(routes._MAX_STORED_COMICS):
        routes.register_comic([{"panel_number": 2}])

    assert routes.get_comic(first) is None
    assert len(routes._COMIC_STORE) == routes._MAX_STORED_COMICS
