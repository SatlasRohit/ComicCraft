import re

import pytest
from PIL import Image

import app.exporters as exporters


@pytest.fixture(autouse=True)
def export_dir(tmp_path, monkeypatch):
    directory = tmp_path / "exports"
    monkeypatch.setenv("PDF_OUTPUT_DIR", str(directory))
    return directory


def make_layout(tmp_path):
    layout = []
    for i in range(1, 6):
        image_path = tmp_path / f"panel_{i}.png"
        Image.new("RGB", (64, 48), (90 + i * 10, 120, 200)).save(image_path, format="PNG")
        layout.append(
            {
                "panel_number": i,
                "title": f"Panel Title {i}",
                "scene_description": f"Scene description {i}",
                "narration": f"Narration {i} — the fox ventures deeper.",
                "dialogue": f"Dialogue line {i}",
                "image_prompt": f"image prompt {i}",
                "image_path": str(image_path),
                "image_url": f"/static/panels/panel_{i}.png",
            }
        )
    return layout


def test_save_pdf_creates_valid_file(tmp_path, export_dir):
    result = exporters.save_pdf(make_layout(tmp_path))

    assert result.startswith(str(export_dir))
    assert result.endswith(".pdf")
    content = open(result, "rb").read()
    assert content[:5] == b"%PDF-"
    assert len(content) > 0


def test_save_pdf_uses_unique_timestamped_filenames(tmp_path, export_dir):
    first = exporters.save_pdf(make_layout(tmp_path))
    second = exporters.save_pdf(make_layout(tmp_path))

    assert first != second
    for name in (first, second):
        assert re.search(r"comic_\d{8}_\d{6}_[0-9a-f]{8}\.pdf$", name)


def test_save_pdf_contains_five_pages(tmp_path, export_dir):
    result = exporters.save_pdf(make_layout(tmp_path))
    content = open(result, "rb").read()

    assert len(re.findall(rb"/Type /Page\b", content)) == 5


def test_save_pdf_with_missing_image_still_succeeds(tmp_path, export_dir):
    layout = make_layout(tmp_path)
    layout[2]["image_path"] = str(tmp_path / "does_not_exist.png")

    result = exporters.save_pdf(layout)

    assert open(result, "rb").read()[:5] == b"%PDF-"


def test_save_pdf_with_corrupt_image_still_succeeds(tmp_path, export_dir):
    layout = make_layout(tmp_path)
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"this is not a png at all")
    layout[1]["image_path"] = str(corrupt)

    result = exporters.save_pdf(layout)

    assert open(result, "rb").read()[:5] == b"%PDF-"


def test_save_pdf_rejects_layout_with_missing_field(tmp_path, export_dir):
    layout = make_layout(tmp_path)
    del layout[0]["dialogue"]

    with pytest.raises(ValueError, match="panel 1 is missing fields"):
        exporters.save_pdf(layout)


def test_save_pdf_rejects_empty_layout(export_dir):
    with pytest.raises(ValueError, match="non-empty list"):
        exporters.save_pdf([])


def test_save_pdf_rejects_non_list(export_dir):
    with pytest.raises(ValueError, match="non-empty list"):
        exporters.save_pdf({"panel_number": 1})


def test_export_dir_honors_env_override(export_dir):
    assert exporters.export_dir() == export_dir


def test_latin1_sanitizes_unicode(export_dir):
    text = exporters._latin1("curly \u2018quotes\u2019 and \u2014 dash")

    assert text == "curly 'quotes' and - dash"
    text.encode("latin-1")
