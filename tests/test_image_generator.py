from pathlib import Path
from types import SimpleNamespace

import pytest

import app.image_generator as image_generator


class FakeImage:
    def save(self, path):
        Path(path).write_bytes(b"fake-png-bytes")


class FakePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[FakeImage()])


SD_ENV_VARS = [
    "SD_MODEL_ID",
    "SD_DEVICE",
    "SD_NUM_INFERENCE_STEPS",
    "SD_GUIDANCE_SCALE",
    "SD_WIDTH",
    "SD_HEIGHT",
    "SD_SEED",
    "SD_OUTPUT_DIR",
    "SD_NEGATIVE_PROMPT",
    "SD_ENABLE_ATTENTION_SLICING",
    "SD_ENABLE_CPU_OFFLOAD",
    "SD_SAFETY_CHECKER",
    "HF_API_KEY",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in SD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SD_DEVICE", "cpu")


def test_load_config_defaults():
    config = image_generator.load_config()

    assert config.model_id == image_generator.DEFAULT_MODEL_ID
    assert config.device == "cpu"
    assert config.num_inference_steps == 25
    assert config.guidance_scale == 7.5
    assert config.width == 512 and config.height == 512
    assert config.seed is None
    assert config.output_dir == Path("static/panels")
    assert config.token is None


def test_load_config_reads_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SD_MODEL_ID", "test/model")
    monkeypatch.setenv("SD_NUM_INFERENCE_STEPS", "7")
    monkeypatch.setenv("SD_GUIDANCE_SCALE", "3.5")
    monkeypatch.setenv("SD_WIDTH", "256")
    monkeypatch.setenv("SD_HEIGHT", "320")
    monkeypatch.setenv("SD_SEED", "42")
    monkeypatch.setenv("SD_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("SD_ENABLE_CPU_OFFLOAD", "true")
    monkeypatch.setenv("HF_API_KEY", "hf_test")

    config = image_generator.load_config()

    assert config.model_id == "test/model"
    assert config.num_inference_steps == 7
    assert config.guidance_scale == 3.5
    assert config.width == 256 and config.height == 320
    assert config.seed == 42
    assert config.output_dir == tmp_path
    assert config.enable_cpu_offload is True
    assert config.token == "hf_test"


def test_build_output_path_uses_predictable_filenames(tmp_path):
    path = image_generator.build_output_path(3, tmp_path)

    assert path == tmp_path / "panel_3.png"
    assert tmp_path.is_dir()


@pytest.mark.parametrize("panel_number", [0, 6, -1])
def test_build_output_path_rejects_invalid_panel(panel_number, tmp_path):
    with pytest.raises(ValueError):
        image_generator.build_output_path(panel_number, tmp_path)


def test_pipeline_is_loaded_once(monkeypatch):
    image_generator.reset_pipeline()
    build_calls = []

    def fake_build(config):
        build_calls.append(config)
        return object()

    monkeypatch.setattr(image_generator, "_build_pipeline", fake_build)
    try:
        config = image_generator.load_config()
        first = image_generator.get_pipeline(config)
        second = image_generator.get_pipeline(config)

        assert first is second
        assert len(build_calls) == 1
    finally:
        image_generator.reset_pipeline()


def test_generate_image_saves_expected_panel_file(monkeypatch, tmp_path):
    monkeypatch.setenv("SD_OUTPUT_DIR", str(tmp_path))
    fake_pipeline = FakePipeline()
    monkeypatch.setattr(image_generator, "get_pipeline", lambda config=None: fake_pipeline)

    result = image_generator.generate_image("a brave fox in a forest", 2)

    assert result == str(tmp_path / "panel_2.png")
    assert (tmp_path / "panel_2.png").read_bytes() == b"fake-png-bytes"
    assert fake_pipeline.calls[0]["prompt"] == "a brave fox in a forest"


def test_generate_image_rejects_empty_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv("SD_OUTPUT_DIR", str(tmp_path))

    with pytest.raises(ValueError):
        image_generator.generate_image("   ", 1)


def test_generate_image_rejects_invalid_panel(monkeypatch, tmp_path):
    monkeypatch.setenv("SD_OUTPUT_DIR", str(tmp_path))

    with pytest.raises(ValueError):
        image_generator.generate_image("a scene", 9)


def test_detect_device_returns_string():
    assert image_generator.detect_device() in {"cpu", "cuda", "mps"}
