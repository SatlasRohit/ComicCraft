from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_export_success_shows_download_link(tmp_path):
    exports = tmp_path / "exports"
    exports.mkdir()
    (exports / "comic_123.pdf").write_bytes(b"%PDF-fake")

    response = client.get("/export-success", params={"pdf": "comic_123.pdf"})

    assert response.status_code == 200
    assert 'href="/static/exports/comic_123.pdf"' in response.text
    assert "download" in response.text
    assert "Go Create Another Comic" in response.text


def test_export_success_without_pdf_param():
    response = client.get("/export-success")

    assert response.status_code == 200
    assert "/static/exports/" not in response.text
    assert "Go Create Another Comic" in response.text


def test_export_success_sanitizes_traversal_attempts():
    response = client.get("/export-success", params={"pdf": "../../secret.pdf"})

    assert response.status_code == 200
    assert "../" not in response.text
    assert 'href="/static/exports/secret.pdf"' in response.text
