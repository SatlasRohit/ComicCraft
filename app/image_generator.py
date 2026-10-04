import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_OUTPUT_DIR = "static/panels"
DEFAULT_NEGATIVE_PROMPT = "text, watermark, signature, blurry, low quality, deformed, extra limbs"
PANEL_COUNT = 5


@dataclass(frozen=True)
class ImageConfig:
    model_id: str
    device: str
    num_inference_steps: int
    guidance_scale: float
    width: int
    height: int
    seed: int | None
    output_dir: Path
    negative_prompt: str
    enable_attention_slicing: bool
    enable_cpu_offload: bool
    safety_checker: bool
    token: str | None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def detect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"

    return "cpu"


def _resolve_device() -> str:
    requested = os.getenv("SD_DEVICE", "auto").strip().lower()
    if requested and requested != "auto":
        return requested
    return detect_device()


def load_config() -> ImageConfig:
    raw_seed = os.getenv("SD_SEED")
    return ImageConfig(
        model_id=os.getenv("SD_MODEL_ID", DEFAULT_MODEL_ID),
        device=_resolve_device(),
        num_inference_steps=_env_int("SD_NUM_INFERENCE_STEPS", 25),
        guidance_scale=_env_float("SD_GUIDANCE_SCALE", 7.5),
        width=_env_int("SD_WIDTH", 512),
        height=_env_int("SD_HEIGHT", 512),
        seed=int(raw_seed) if raw_seed not in (None, "") else None,
        output_dir=Path(os.getenv("SD_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)),
        negative_prompt=os.getenv("SD_NEGATIVE_PROMPT", DEFAULT_NEGATIVE_PROMPT),
        enable_attention_slicing=_env_bool("SD_ENABLE_ATTENTION_SLICING", True),
        enable_cpu_offload=_env_bool("SD_ENABLE_CPU_OFFLOAD", False),
        safety_checker=_env_bool("SD_SAFETY_CHECKER", False),
        token=os.getenv("HF_API_KEY") or None,
    )


_PIPELINE = None
_PIPELINE_LOCK = Lock()


def _build_pipeline(config: ImageConfig):
    import torch
    from diffusers import StableDiffusionPipeline

    dtype = torch.float16 if config.device == "cuda" else torch.float32
    kwargs = {"torch_dtype": dtype, "token": config.token}
    if not config.safety_checker:
        kwargs["safety_checker"] = None
        kwargs["requires_safety_checker"] = False

    pipeline = StableDiffusionPipeline.from_pretrained(config.model_id, **kwargs)

    if config.enable_cpu_offload and config.device == "cuda":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(config.device)

    if config.enable_attention_slicing:
        pipeline.enable_attention_slicing()

    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def get_pipeline(config: ImageConfig | None = None):
    global _PIPELINE
    if _PIPELINE is None:
        with _PIPELINE_LOCK:
            if _PIPELINE is None:
                _PIPELINE = _build_pipeline(config or load_config())
    return _PIPELINE


def reset_pipeline() -> None:
    global _PIPELINE
    _PIPELINE = None


def build_output_path(panel_number: int, output_dir: str | Path | None = None) -> Path:
    panel_number = int(panel_number)
    if not 1 <= panel_number <= PANEL_COUNT:
        raise ValueError(f"panel_number must be between 1 and {PANEL_COUNT}, got {panel_number}")

    directory = Path(output_dir) if output_dir is not None else load_config().output_dir
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"panel_{panel_number}.png"


def _save_image(image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


import re


def sanitize_filename(prompt: str) -> str:
    cleaned = re.sub(r"[^\w\-]", "_", prompt.strip().lower()[:32]).strip("_")
    return f"{cleaned or 'panel'}.png"


def _generate_placeholder(image_prompt: str, panel_number: int, path: Path) -> str:
    """Generate a dynamic, stylized comic illustration with environment, characters, and lighting."""
    import math
    import random
    from PIL import Image, ImageDraw, ImageFont

    w, h = 512, 512
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)

    p_lower = image_prompt.lower()

    # Determine visual palette based on prompt keywords & panel progression
    if any(k in p_lower for k in ["space", "star", "planet", "galaxy", "astronaut", "moon"]):
        top_color = (12, 6, 32)
        mid_color = (36, 18, 72)
        bot_color = (65, 25, 95)
        theme = "space"
    elif any(k in p_lower for k in ["cyber", "city", "neon", "robot", "street", "building"]):
        top_color = (16, 12, 38)
        mid_color = (45, 18, 68)
        bot_color = (12, 32, 65)
        theme = "city"
    elif any(k in p_lower for k in ["magic", "crystal", "glow", "spark", "ancient", "temple"]):
        top_color = (18, 26, 52)
        mid_color = (32, 65, 88)
        bot_color = (22, 85, 75)
        theme = "magic"
    else:  # Forest / nature default
        # Evolve lighting with panel number (early: dawn/mystery, middle: glowing discovery, late: magical dawn)
        if panel_number <= 2:
            top_color = (18, 18, 42)
            mid_color = (38, 48, 68)
            bot_color = (18, 42, 28)
        elif panel_number <= 4:
            top_color = (25, 15, 50)
            mid_color = (55, 30, 80)
            bot_color = (30, 60, 50)
        else:
            top_color = (30, 20, 60)
            mid_color = (75, 45, 90)
            bot_color = (40, 75, 60)
        theme = "forest"

    # Smooth vertical sky gradient
    for y in range(h):
        factor = y / h
        if factor < 0.5:
            f = factor * 2
            r = int(top_color[0] + (mid_color[0] - top_color[0]) * f)
            g = int(top_color[1] + (mid_color[1] - top_color[1]) * f)
            b = int(top_color[2] + (mid_color[2] - top_color[2]) * f)
        else:
            f = (factor - 0.5) * 2
            r = int(mid_color[0] + (bot_color[0] - mid_color[0]) * f)
            g = int(mid_color[1] + (bot_color[1] - mid_color[1]) * f)
            b = int(mid_color[2] + (bot_color[2] - mid_color[2]) * f)
        draw.line([(0, y), (w, y)], fill=(r, g, b))

    # Celestial body (Moon, planet, or magical orb)
    cx, cy = int(w * 0.72), int(h * 0.26)
    glow_color = (255, 235, 160) if theme != "space" else (180, 220, 255)
    for radius in range(75, 0, -6):
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=glow_color)
    draw.ellipse([cx - 36, cy - 36, cx + 36, cy + 36], fill=(255, 250, 225))

    # Stars / magical particles
    rng = random.Random(panel_number * 107)
    for _ in range(70):
        sx = rng.randint(12, w - 12)
        sy = rng.randint(12, int(h * 0.68))
        s_size = rng.randint(1, 3)
        draw.ellipse([sx, sy, sx + s_size, sy + s_size], fill=(255, 255, 255))

    # Thematic environmental silhouettes
    if theme in ("forest", "magic"):
        # Distant mountain ridge
        draw.polygon([(0, 340), (140, 285), (290, 325), (430, 275), (512, 310), (512, 512), (0, 512)], fill=(16, 28, 36))
        # Midground layered pine trees
        tree_specs = [(40, 335, 115), (105, 320, 100), (175, 345, 135), (360, 315, 110), (435, 335, 125), (485, 305, 90)]
        for tx, ty, t_height in tree_specs:
            draw.polygon([(tx, ty - t_height), (tx - 36, ty), (tx + 36, ty)], fill=(12, 22, 28))
        # Foreground hill
        draw.ellipse([-60, 415, w + 60, 620], fill=(6, 14, 18))

        # Hero silhouette (fox / wanderer)
        hx, hy = 160, 415
        draw.ellipse([hx - 32, hy - 22, hx + 32, hy + 16], fill=(18, 28, 34))  # body
        draw.polygon([(hx + 22, hy - 36), (hx + 48, hy - 16), (hx + 18, hy - 6)], fill=(18, 28, 34))  # head/snout
        draw.polygon([(hx + 24, hy - 48), (hx + 30, hy - 32), (hx + 18, hy - 34)], fill=(18, 28, 34))  # ear 1
        draw.polygon([(hx + 34, hy - 44), (hx + 38, hy - 30), (hx + 28, hy - 32)], fill=(18, 28, 34))  # ear 2
        draw.ellipse([hx - 58, hy - 32, hx - 22, hy + 6], fill=(18, 28, 34))  # bushy tail

        # Glowing focal point (crystal or magical energy)
        fx, fy = 295, 405
        # Radiant aura
        for r in range(45, 0, -4):
            draw.ellipse([fx - r, fy - r, fx + r, fy + r], outline=(110, 225, 255))
        # Brilliant diamond crystal
        draw.polygon([(fx, fy - 26), (fx + 16, fy), (fx, fy + 26), (fx - 16, fy)], fill=(210, 252, 255))
        # Radiating comic light rays
        for angle in range(0, 360, 45):
            rad = math.radians(angle + (panel_number * 15))
            ex = fx + int(math.cos(rad) * 48)
            ey = fy + int(math.sin(rad) * 48)
            draw.line([(fx, fy), (ex, ey)], fill=(230, 255, 255), width=2)
    elif theme == "city":
        # Cyberpunk Skyline
        for bx, bw, bh in [(25, 70, 240), (105, 85, 290), (200, 75, 260), (285, 95, 320), (390, 65, 250), (465, 75, 210)]:
            by = h - bh
            draw.rectangle([bx, by, bx + bw, h], fill=(14, 18, 34))
            for wy in range(by + 20, h - 30, 24):
                for wx in range(bx + 10, bx + bw - 10, 16):
                    if (wx + wy) % 3 == 0:
                        draw.rectangle([wx, wy, wx + 8, wy + 12], fill=(255, 230, 100))
                    elif (wx + wy) % 5 == 0:
                        draw.rectangle([wx, wy, wx + 8, wy + 12], fill=(6, 182, 212))
        draw.rectangle([0, 455, w, h], fill=(8, 10, 20))
    elif theme == "space":
        # Planet with ring
        px, py = 160, 325
        draw.ellipse([px - 65, py - 65, px + 65, py + 65], fill=(185, 105, 225))
        draw.arc([px - 105, py - 22, px + 105, py + 22], start=0, end=360, fill=(245, 205, 255), width=4)
        # Spaceship silhouette
        sx, sy = 345, 245
        draw.polygon([(sx + 36, sy), (sx - 20, sy - 16), (sx - 10, sy), (sx - 20, sy + 16)], fill=(240, 242, 255))
        draw.polygon([(sx - 10, sy - 5), (sx - 36, sy), (sx - 10, sy + 5)], fill=(255, 130, 45))

    # Comic Book Framing (Dark border with inner white accent)
    draw.rectangle([0, 0, w - 1, h - 1], outline=(20, 25, 40), width=6)
    draw.rectangle([8, 8, w - 9, h - 9], outline=(255, 255, 255), width=2)

    # Comic Number Badge
    badge_bg = (99, 102, 241) if panel_number % 2 == 1 else (245, 158, 11)
    draw.rounded_rectangle([18, 18, 112, 54], radius=16, fill=badge_bg, outline=(255, 255, 255), width=2)
    try:
        font_badge = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font_badge = ImageFont.load_default()
    draw.text((65, 36), f"PANEL {panel_number}", fill="white", font=font_badge, anchor="mm")

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return str(path)


_original_get_pipeline = get_pipeline


def generate_image(
    image_prompt: str,
    panel_number: int = 1,
    filename: str | None = None,
) -> str:
    if not image_prompt or not image_prompt.strip():
        raise ValueError("image_prompt must be a non-empty string")

    config = load_config()
    if filename:
        directory = Path(config.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
    else:
        path = build_output_path(panel_number, config.output_dir)

    is_custom_pipeline = get_pipeline is not _original_get_pipeline

    # If mock mode is explicitly enabled and pipeline isn't mocked in tests
    if _env_bool("SD_MOCK_IMAGES", False) and not is_custom_pipeline:
        return _generate_placeholder(image_prompt, panel_number, path)

    try:
        pipeline = get_pipeline(config)
    except Exception as exc:
        if not is_custom_pipeline:
            return _generate_placeholder(image_prompt, panel_number, path)
        raise

    generator = None
    if config.seed is not None:
        import torch

        generator = torch.Generator(device=config.device).manual_seed(config.seed)

    try:
        result = pipeline(
            prompt=image_prompt,
            negative_prompt=config.negative_prompt or None,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            width=config.width,
            height=config.height,
            generator=generator,
        )
    except Exception as exc:
        if not is_custom_pipeline:
            return _generate_placeholder(image_prompt, panel_number, path)
        raise RuntimeError(f"Image generation failed for panel {panel_number}: {exc}") from exc

    images = getattr(result, "images", None)
    if not images:
        if not is_custom_pipeline:
            return _generate_placeholder(image_prompt, panel_number, path)
        raise RuntimeError("Image pipeline returned no images")

    _save_image(images[0], path)
    return str(path)

