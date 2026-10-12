import json
from pathlib import Path

import pytest

from src.domain.animation import AnimationSpec
from src.rendering import skeleton as sk
from src.rendering.runner import extract_still, render_animation, render_frame

SAMPLES = Path("samples/specs")
INDEX = json.loads((SAMPLES / "index.json").read_text(encoding="utf-8"))


def load(entry: dict) -> AnimationSpec:
    return AnimationSpec.model_validate(json.loads((SAMPLES / entry["file"]).read_text(encoding="utf-8")))


@pytest.mark.slow
def test_real_manim_render_of_a_sample(tmp_path):
    # Arrange
    entry = INDEX[0]
    spec = load(entry)
    video = tmp_path / "out.mp4"
    image = tmp_path / "out.png"

    # Act
    render_animation(spec, entry["title"], entry["equipment"], ["hamstrings", "glutes"], video)
    extract_still(video, image, sk.peak_activation_time(spec) + 0.3)

    # Assert
    assert video.stat().st_size > 10_000
    assert image.stat().st_size > 1_000


@pytest.mark.slow
@pytest.mark.parametrize("entry", INDEX, ids=lambda e: e["file"])
def test_every_sample_renders_a_frame(tmp_path, entry):
    # Arrange
    spec = load(entry)
    image = tmp_path / "frame.png"

    # Act
    render_frame(spec, entry["title"], entry["equipment"], [], image, sk.peak_activation_time(spec))

    # Assert
    assert image.stat().st_size > 1_000


@pytest.mark.slow
def test_storyboard_renders_one_tile_per_key_pose(tmp_path):
    # Arrange
    from PIL import Image

    from src.rendering.storyboard import TILE_SIZE, render_storyboard, storyboard_times

    entry = next(e for e in INDEX if e["file"] == "bench_press_barbell.json")
    spec = load(entry)

    # Act
    image = render_storyboard(spec, entry["title"], entry["equipment"], [], tmp_path / "board.png")

    # Assert
    assert Image.open(image).size == (TILE_SIZE[0] * len(storyboard_times(spec)), TILE_SIZE[1])
