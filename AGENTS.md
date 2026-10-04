# AGENTS.md — ComicCraft

## What this is
Web app that turns a user prompt into a 5-panel AI comic (story + illustrations) and exports it as a PDF.
Stack: Python + FastAPI (backend), Jinja2/HTML/CSS (frontend), Google Gemini (story text), Hugging Face Diffusers / Stable Diffusion (panel images), FPDF (PDF export).

The full spec lives in `ComicCraft - AI Comic Story Creator using Gemini Models.docx.pdf` at the repo root. If prose and code disagree, trust the code.

## Status
All modules and routes are implemented and tested (AI calls mocked in tests). A real run needs `GEMINI_API_KEY` in `.env`; image generation downloads the Stable Diffusion model on first use. The comic store is in-memory only — it does not survive server restarts.

## Architecture
Strict separation — one concern per module, don't merge them:

| Module | Responsibility |
| --- | --- |
| `app/main.py` | FastAPI app, mounts `/static`, includes router |
| `app/routes.py` | HTTP routes, form/JSON input capture, orchestration |
| `app/gemini_flash.py` | `generate_outline()` → structured 5-panel outline |
| `app/gemini_pro.py` | `generate_story()` → narration + dialogue per panel |
| `app/image_generator.py` | `generate_image()` → Stable Diffusion panel image |
| `app/layout_builder.py` | `build_comic_layout()` → merge image + text per panel |
| `app/exporters.py` | `save_pdf()` → multi-page PDF via FPDF |

Do not introduce new frameworks, ORMs, queues, or frontend build tools. Use the spec's stack only.

## Directory structure
```
app/            main, routes, gemini_flash, gemini_pro, image_generator, layout_builder, exporters
templates/      index.html, comic_preview.html, export_success.html
static/panels/  generated panel images (runtime output)
static/exports/ generated PDF files (runtime output)
static/fonts/   PDF fonts
static/css/     shared stylesheet (styles.css) — all three templates link it; do not re-embed per-page CSS
.env            secrets (gitignored — never commit)
requirements.txt
README.md
AGENTS.md
```

Template names are `index.html`, `comic_preview.html`, `export_success.html`. Ignore the spec's stray `result.html` / `all_users.html` mentions — there is no account/admin feature.

## AI workflow (order matters)
1. `generate_outline()` (Gemini Flash) — 5 panels: number, title, scene description, image prompt.
2. `generate_story()` (Gemini Pro) — narration and character dialogue per panel.
3. `generate_image()` (Stable Diffusion) — one image per panel.
4. `build_comic_layout()` — combine panel image, title, scene description, narration, dialogue.
5. `save_pdf()` — compile layout into a timestamped PDF under `static/exports/`.

Pipeline entrypoints: `POST /generate` (HTML form) and `POST /generate-comic/json` (JSON via `PromptRequest`). `GET /` renders `index.html`, `GET /export-success` renders `export_success.html`, `/test-image` is a dev-only image utility. `POST /export` (form with hidden `comic_id`) runs `save_pdf()` on the stored layout and redirects to `/export-success`.

State: `routes.py` keeps an in-memory comic store (`register_comic()` / `get_comic()`); `/generate` and `/generate-comic/json` register the layout, and the preview template's export forms carry the hidden `comic_id`. Form/JSON field names are `story_prompt`, `character_name`, `setting`, `tone`, `art_style` (the index form field is `name="story_prompt"`, not `prompt`).

## Conventions
- FastAPI routes live only in `app/routes.py`; keep AI/PDF logic out of routes.
- Forms use `Form(...)` params; JSON uses a Pydantic `PromptRequest` model.
- Templates rendered with `Jinja2Templates(directory="templates")` and `TemplateResponse`.
- Gemtext output (outlines) must be parsed into a list of dicts; never pass raw model text into templates expecting structure.
- Prompt-derived filenames must be sanitized before saving to `static/panels/`.
- Store generated artifacts only under `static/panels/` and `static/exports/`.
- Do not hand-edit generated images, PDFs, or other runtime output — regenerate them instead.

## Secrets
- Never hardcode API keys or tokens.
- Load from `.env` via `python-dotenv`; expected vars: `GEMINI_API_KEY`, `HF_API_KEY`.
- Commit `.env.example` only; `.env` stays in `.gitignore`.

## Testing
- Run the app: `uvicorn app.main:app --reload` → http://127.0.0.1:8000 (API docs at `/docs`).
- Run tests: `pytest` (config in `pytest.ini`). Unit tests mock the Gemini client — no API key needed.
- `tests/test_gemini_live.py` is skipped unless `GEMINI_API_KEY` is set; it makes a real API call.
- On machines with ROS sourced, `PYTHONPATH` pulls in ROS pytest plugins; `pytest.ini` `addopts` disables them.
- Test each module in isolation before wiring it into `/generate` (outline → story → image → layout → PDF).
- Focused checks: use `/test-image` for image generation; `/docs` to exercise JSON routes.
- After any change, verify the affected module and, once integrated, the full workflow (prompt → preview → PDF).
- Image generation may be slow on CPU; note whether CUDA is available before assuming runtimes.
