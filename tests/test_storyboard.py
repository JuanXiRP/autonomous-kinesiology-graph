import json
from pathlib import Path

import pytest
from PIL import Image

from src.domain.animation import AnimationSpec
from src.rendering import skeleton as sk
from src.rendering import storyboard

SAMPLES = Path("samples/specs")
INDEX = json.loads((SAMPLES / "index.json").read_text(encoding="utf-8"))


def load(name: str) -> AnimationSpec:
    return AnimationSpec.model_validate(json.loads((SAMPLES / name).read_text(encoding="utf-8")))


@pytest.mark.parametrize("entry", INDEX, ids=lambda e: e["file"])
def test_storyboard_shows_distinct_key_poses_of_the_first_rep(entry):
    # Arrange
    spec = load(entry["file"])

    # Act
    times = storyboard.storyboard_times(spec)

    # Assert
    assert 2 <= len(times) <= storyboard.MAX_TILES
    assert times == sorted(set(times))
    assert times[-1] < sk.build_timeline(spec).rep_duration


def test_long_specs_are_sampled_down_to_the_tile_limit():
    # Arrange
    data = json.loads((SAMPLES / "romanian_deadlift_barbell.json").read_text(encoding="utf-8"))
    middle = data["keyframes"][1]
    data["keyframes"] = data["keyframes"][:1] + [middle] * 6 + data["keyframes"][-1:]
    spec = AnimationSpec.model_validate(data)

    # Act
    times = storyboard.storyboard_times(spec)

    # Assert
    assert len(times) == storyboard.MAX_TILES
    assert times[0] == sk.build_timeline(spec).arrive[0] + spec.keyframes[0].hold_s / 2


def test_compose_tiles_left_to_right():
    # Arrange
    tiles = [Image.new("RGB", (1280, 720), color) for color in ("red", "blue")]

    # Act
    sheet = storyboard.compose(tiles)

    # Assert
    assert sheet.size == (storyboard.TILE_SIZE[0] * 2, storyboard.TILE_SIZE[1])
    assert sheet.getpixel((10, 10)) == (255, 0, 0)
    assert sheet.getpixel((storyboard.TILE_SIZE[0] + 10, 10)) == (0, 0, 255)
