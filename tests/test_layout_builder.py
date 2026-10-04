import copy

import pytest

from app.layout_builder import PANEL_COUNT, build_comic_layout


def make_story():
    return [
        {
            "panel_number": i,
            "title": f"Panel {i}",
            "scene_description": f"Scene {i}",
            "narration": f"Narration {i}",
            "dialogue": f"Dialogue {i}",
            "image_prompt": f"image prompt {i}",
        }
        for i in range(1, PANEL_COUNT + 1)
    ]


@pytest.fixture
def story():
    return make_story()


@pytest.fixture
def image_files(tmp_path):
    files = []
    for i in range(1, PANEL_COUNT + 1):
        path = tmp_path / f"panel_{i}.png"
        path.write_bytes(b"png")
        files.append(str(path))
    return files


def test_five_valid_panels(story, image_files):
    layout = build_comic_layout(story, image_files)

    assert len(layout) == PANEL_COUNT
    assert [p["panel_number"] for p in layout] == [1, 2, 3, 4, 5]
    assert set(layout[0]) == {
        "panel_number",
        "title",
        "scene_description",
        "narration",
        "dialogue",
        "image_prompt",
        "image_path",
        "image_url",
    }
    assert layout[2]["image_path"] == image_files[2]
    assert layout[0]["title"] == "Panel 1"


def test_valid_panels_accept_panel_number_dict(story, image_files):
    paths_by_panel = {i: path for i, path in enumerate(image_files, start=1)}

    layout = build_comic_layout(story, paths_by_panel)

    assert [p["panel_number"] for p in layout] == [1, 2, 3, 4, 5]
    assert layout[4]["image_path"] == paths_by_panel[5]


def test_image_url_points_at_static_mount(story, tmp_path):
    panels_dir = tmp_path / "static" / "panels"
    panels_dir.mkdir(parents=True)
    files = []
    for i in range(1, PANEL_COUNT + 1):
        path = panels_dir / f"panel_{i}.png"
        path.write_bytes(b"png")
        files.append(str(path))

    layout = build_comic_layout(story, files)

    assert layout[0]["image_path"].endswith("panel_1.png")
    assert layout[0]["image_url"] == "/static/panels/panel_1.png"
    assert layout[4]["image_url"] == "/static/panels/panel_5.png"


def test_missing_image_file_raises(story, image_files):
    import os

    os.remove(image_files[2])

    with pytest.raises(ValueError, match="panel 3.*does not exist"):
        build_comic_layout(story, image_files)


def test_missing_image_path_in_dict_raises(story, image_files):
    paths_by_panel = {i: path for i, path in enumerate(image_files, start=1)}
    del paths_by_panel[3]

    with pytest.raises(ValueError, match=r"missing image paths for panels: \[3\]"):
        build_comic_layout(story, paths_by_panel)


def test_short_image_path_list_raises(story, image_files):
    with pytest.raises(ValueError, match=r"missing image paths for panels: \[5\]"):
        build_comic_layout(story, image_files[:4])


def test_incomplete_panel_data_raises(story, image_files):
    broken = copy.deepcopy(story)
    del broken[1]["narration"]

    with pytest.raises(ValueError, match="panel 2 is missing required story fields: narration"):
        build_comic_layout(broken, image_files)


def test_empty_story_field_raises(story, image_files):
    broken = copy.deepcopy(story)
    broken[4]["dialogue"] = ""

    with pytest.raises(ValueError, match="panel 5 is missing required story fields: dialogue"):
        build_comic_layout(broken, image_files)


def test_wrong_panel_count_raises(image_files):
    with pytest.raises(ValueError, match="exactly 5 panels, got 4"):
        build_comic_layout(make_story()[:4], image_files)


def test_out_of_order_panel_numbers_raises(story, image_files):
    broken = copy.deepcopy(story)
    broken[0]["panel_number"] = 2

    with pytest.raises(ValueError, match="numbered 1..5 in order"):
        build_comic_layout(broken, image_files)


def test_invalid_image_paths_type_raises(story):
    with pytest.raises(ValueError, match="image_paths must be"):
        build_comic_layout(story, "static/panels/panel_1.png")
