import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.routes as routes
from app.main import app
from app.exporters import export_dir


@pytest.fixture(autouse=True)
def exports_tmp(tmp_path, monkeypatch):
    directory = tmp_path / "exports"
    monkeypatch.setenv("PDF_OUTPUT_DIR", str(directory))
    return directory


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_layout(tmp_path):
    layout = []
    for i in range(1, 6):
        image_path = tmp_path / f"panel_{i}.png"
        Image.new("RGB", (64, 48), (100, 140, 220)).save(image_path, format="PNG")
        layout.append(
            {
                "panel_number": i,
                "title": f"Panel Title {i}",
                "scene_description": f"Scene description {i}",
                "narration": f"Narration {i}",
                "dialogue": f"Dialogue line {i}",
                "image_prompt": f"image prompt {i}",
                "image_path": str(image_path),
                "image_url": f"/static/panels/panel_{i}.png",
            }
        )
    return layout


def test_register_and_get_comic(mock_layout):
    comic_id = routes.register_comic(mock_layout)

    assert routes.get_comic(comic_id) == mock_layout
    assert routes.get_comic("missing") is None


def test_export_unknown_comic_returns_404(client):
    response = client.post("/export", data={"comic_id": "does-not-exist"})

    assert response.status_code == 404


def test_export_creates_pdf_and_redirects_to_success(client, mock_layout):
    comic_id = routes.register_comic(mock_layout)

    response = client.post("/export", data={"comic_id": comic_id}, follow_redirects=True)

    assert response.status_code == 200
    assert "/export-success" in str(response.url)
    assert "Download PDF" in response.text
    assert "/static/exports/comic_" in response.text

    exported = list(export_dir().glob("comic_*.pdf"))
    assert len(exported) == 1
