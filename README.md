# ComicCraft — AI Comic Story Creator using Gemini Models

ComicCraft turns a user prompt into a personalized 5-panel AI comic (story + illustrations) and exports it as a PDF.

## Stack
- **Backend:** Python, FastAPI, Uvicorn
- **Frontend:** HTML, CSS, Jinja2 templates
- **Story AI:** Google Gemini (Flash for the 5-panel outline, Pro for narration/dialogue)
- **Image AI:** Hugging Face Diffusers / Stable Diffusion
- **PDF export:** FPDF

## Setup
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `GEMINI_API_KEY` and `HF_API_KEY` in `.env`. Never commit `.env`.

The story models default to Google's current stable aliases (`gemini-flash-latest` for the outline,
`gemini-pro-latest` for narration/dialogue). If a model is retired or your plan lacks access, override
with `GEMINI_FLASH_MODEL` / `GEMINI_PRO_MODEL` in `.env` — e.g. `GEMINI_PRO_MODEL=gemini-2.5-flash`
if your quota does not include pro-tier models.

## Run
```bash
uvicorn app.main:app --reload
```
- App: http://127.0.0.1:8000
- API docs: http://127.0.0.1:8000/docs

## Project structure
```
app/
    main.py            FastAPI app, mounts /static, includes router
    routes.py          HTTP routes and orchestration
    gemini_flash.py    generate_outline()
    gemini_pro.py      generate_story()
    image_generator.py generate_image()
    layout_builder.py  build_comic_layout()
    exporters.py       save_pdf()
templates/
    index.html
    comic_preview.html
    export_success.html
static/
    css/               shared stylesheet (styles.css)
    panels/            generated panel images
    exports/           generated PDFs
    fonts/             PDF fonts
tests/
```

## Workflow
1. User submits prompt, character name, setting, tone, and art style.
2. `generate_outline()` builds a structured 5-panel outline (Gemini Flash).
3. `generate_story()` writes narration and dialogue per panel (Gemini Pro).
4. `generate_image()` creates one illustration per panel (Stable Diffusion).
5. `build_comic_layout()` merges image, title, scene description, narration, and dialogue.
6. Comic is previewed and can be exported to PDF via `save_pdf()`.

## Tests
```bash
pytest
```
Unit tests mock the Gemini client, so no API key or network access is needed. `tests/test_gemini_live.py`
is skipped unless `GEMINI_API_KEY` is set, in which case it makes a real Gemini call:

```bash
GEMINI_API_KEY=... pytest tests/test_gemini_live.py
```

## Image generation settings
`image_generator.py` loads the Stable Diffusion pipeline once and reuses it for all panels. It is configured
entirely through environment variables (defaults shown):

| Variable | Default | Purpose |
| --- | --- | --- |
| `SD_MODEL_ID` | `runwayml/stable-diffusion-v1-5` | Diffusers model id |
| `SD_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `SD_NUM_INFERENCE_STEPS` | `25` | Diffusion steps (quality vs speed) |
| `SD_GUIDANCE_SCALE` | `7.5` | Prompt adherence |
| `SD_WIDTH` / `SD_HEIGHT` | `512` | Output resolution |
| `SD_SEED` | unset | Fix the seed for reproducible panels |
| `SD_OUTPUT_DIR` | `static/panels` | Where images are written |
| `SD_ENABLE_ATTENTION_SLICING` | `true` | Lowers VRAM use |
| `SD_ENABLE_CPU_OFFLOAD` | `false` | Offload to CPU on CUDA for low VRAM |
| `SD_SAFETY_CHECKER` | `false` | Enable the NSFW safety checker |

Images are saved with predictable names: `static/panels/panel_1.png` … `panel_5.png`.

### Real image generation test
The first run downloads the model (~4 GB) and needs either a logged-in Hugging Face session or `HF_API_KEY`.
No `/test-image` route exists yet, so call the module directly:

```bash
python -c "from app.image_generator import generate_image; print(generate_image('a brave fox in an enchanted forest, anime style', 1))"
```

## Status
Built phase by phase. **Phase 1 complete:** scaffold + minimal FastAPI app. **Phase 2 complete:** responsive
`index.html` form. **Phase 3 complete:** Gemini story generation (`gemini_flash.py`, `gemini_pro.py`) with
Pydantic-validated JSON output and mocked unit tests. **Phase 4 complete:** cached Stable Diffusion image
generation (`image_generator.py`), configurable via environment variables. **Phase 5 complete:** comic layout
construction (`layout_builder.py`) merging story panels with panel images into a structure for both the
HTML preview (`image_url`) and PDF export (`image_path`). **Phase 6 complete:** responsive comic preview
page (`comic_preview.html`) rendering every panel with number, title, image, scene description, narration,
and dialogue, plus Back / Generate Again / Export PDF actions. **Phase 7 complete:** PDF export
(`exporters.py` via fpdf2, one page per panel, timestamped files under `static/exports/`), `POST /export`
and `GET /export-success` with a download link. **Phase 8 complete:** full pipeline integration —
`POST /generate` (form → preview page), `POST /generate-comic/json` (JSON API returning layout + PDF
path), `GET /test-image` (dev image utility), with per-stage error handling (422/502/503/500) that logs
details server-side and never leaks keys or stack traces to the browser. **Phase 10 complete:** polished
UI — shared design system in `static/css/styles.css` (typography, cards, gradient buttons, shimmer
loading overlay with cycling status messages, speech-bubble dialogue, alternating panel chips,
responsive breakpoints, `prefers-reduced-motion` support) with no frontend frameworks.

### Routes
| Route | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Comic creation form (`index.html`) |
| `/generate` | POST | Form flow: outline → story → images → layout → preview |
| `/generate-comic/json` | POST | JSON API (`PromptRequest`): same flow, returns layout + PDF path |
| `/test-image` | GET | Dev-only: run `generate_image()` on a single prompt |
| `/export` | POST | Compile stored comic (hidden `comic_id`) into a PDF |
| `/export-success` | GET | Success page with download link |
# ComicCraft
