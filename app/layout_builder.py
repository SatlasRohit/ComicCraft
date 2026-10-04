from pathlib import Path

PANEL_COUNT = 5
REQUIRED_STORY_FIELDS = (
    "panel_number",
    "title",
    "scene_description",
    "narration",
    "dialogue",
    "image_prompt",
)


def _validate_story(story) -> None:
    if not isinstance(story, list) or not story:
        raise ValueError("story must be a non-empty list of panel dicts")

    if len(story) != PANEL_COUNT:
        raise ValueError(f"story must contain exactly {PANEL_COUNT} panels, got {len(story)}")

    for index, panel in enumerate(story, start=1):
        if not isinstance(panel, dict):
            raise ValueError(f"panel at index {index} must be a dict, got {type(panel).__name__}")

        missing = [field for field in REQUIRED_STORY_FIELDS if not panel.get(field)]
        if missing:
            raise ValueError(
                f"panel {index} is missing required story fields: {', '.join(missing)}"
            )

        number = panel["panel_number"]
        if number != index:
            raise ValueError(
                f"story panels must be numbered 1..{PANEL_COUNT} in order; "
                f"panel at index {index} has panel_number {number!r}"
            )


def _validate_image_paths(image_paths) -> dict[int, str]:
    if image_paths is None:
        raise ValueError("image_paths is required: provide a path for every panel")

    if isinstance(image_paths, dict):
        try:
            lookup = {int(key): value for key, value in image_paths.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"image_paths dict keys must be panel numbers: {exc}") from exc
    elif isinstance(image_paths, (list, tuple)):
        lookup = {number: path for number, path in enumerate(image_paths, start=1)}
    else:
        raise ValueError(
            "image_paths must be a list of paths ordered by panel, "
            f"or a dict of panel_number -> path; got {type(image_paths).__name__}"
        )

    missing = [number for number in range(1, PANEL_COUNT + 1) if not lookup.get(number)]
    if missing:
        raise ValueError(f"missing image paths for panels: {missing}")

    for number in range(1, PANEL_COUNT + 1):
        path = Path(lookup[number])
        if not path.is_file():
            raise ValueError(f"image file for panel {number} does not exist: {path}")

    return {number: str(lookup[number]) for number in range(1, PANEL_COUNT + 1)}


def _image_url(image_path: str) -> str:
    url = str(image_path).replace("\\", "/")
    static_index = url.find("static/")
    if static_index >= 0:
        return "/" + url[static_index:]
    return url if url.startswith("/") else "/" + url


def build_comic_layout(story: list[dict], image_paths) -> list[dict]:
    _validate_story(story)
    images = _validate_image_paths(image_paths)

    layout = []
    for panel in story:
        number = panel["panel_number"]
        image_path = images[number]
        layout.append(
            {
                "panel_number": number,
                "title": panel["title"],
                "scene_description": panel["scene_description"],
                "narration": panel["narration"],
                "dialogue": panel["dialogue"],
                "image_prompt": panel["image_prompt"],
                "image_path": image_path,
                "image_url": _image_url(image_path),
            }
        )
    return layout
