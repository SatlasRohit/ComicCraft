import pytest
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

PANEL_COUNT = 5


def make_layout():
    return [
        {
            "panel_number": i,
            "title": f"Panel Title {i}",
            "scene_description": f"Scene description {i}",
            "narration": f"Narration {i}",
            "dialogue": f"Dialogue line {i}",
            "image_prompt": f"image prompt {i}",
            "image_path": f"static/panels/panel_{i}.png",
            "image_url": f"/static/panels/panel_{i}.png",
        }
        for i in range(1, PANEL_COUNT + 1)
    ]


def create_client(layout, comic_id=None):
    app = FastAPI()
    app.mount("/static", StaticFiles(directory="static"), name="static")
    templates = Jinja2Templates(directory="templates")

    @app.get("/preview", response_class=HTMLResponse)
    async def preview(request: Request):
        return templates.TemplateResponse(
            request,
            "comic_preview.html",
            {"layout": layout, "comic_id": comic_id},
        )

    return TestClient(app)


@pytest.fixture
def client():
    return create_client(make_layout())


def test_preview_renders_every_panel(client):
    response = client.get("/preview")

    assert response.status_code == 200
    html = response.text
    assert html.count('class="panel"') == PANEL_COUNT
    for i in range(1, PANEL_COUNT + 1):
        assert f"Panel {i}" in html
        assert f"Panel Title {i}" in html
        assert f"Scene description {i}" in html
        assert f"Narration {i}" in html
        assert f"Dialogue line {i}" in html
        assert f"/static/panels/panel_{i}.png" in html


def test_preview_has_action_buttons(client):
    response = client.get("/preview")
    html = response.text

    assert "Back" in html
    assert "Generate Again" in html
    assert "Export PDF" in html
    assert html.count('action="/export"') == 2


def test_preview_links_responsive_stylesheet(client):
    response = client.get("/preview")

    assert response.status_code == 200
    assert 'width=device-width, initial-scale=1.0' in response.text
    assert 'href="/static/css/styles.css"' in response.text

    css = client.get("/static/css/styles.css")
    assert css.status_code == 200
    assert "@media (max-width: 560px)" in css.text
    assert "prefers-reduced-motion" in css.text


def test_preview_export_form_carries_comic_id(client):
    layout = make_layout()
    client = create_client(layout, comic_id="abc123")

    response = client.get("/preview")
    html = response.text

    assert html.count('name="comic_id"') == 2
    assert 'value="abc123"' in html


def test_preview_omits_comic_id_when_absent(client):
    response = client.get("/preview")

    assert 'name="comic_id"' not in response.text


def test_preview_escapes_story_text():
    from markupsafe import escape

    layout = make_layout()
    raw = "<script>alert('comic')</script>"
    layout[0]["title"] = raw
    client = create_client(layout)

    response = client.get("/preview")

    assert response.status_code == 200
    assert raw not in response.text
    assert str(escape(raw)) in response.text
