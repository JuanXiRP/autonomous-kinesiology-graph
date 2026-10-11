import json
from pathlib import Path

import pytest

from src.domain.animation import AnimationSpec
from src.rendering import skeleton as sk
from src.rendering.runner import extract_still, render_animation


@pytest.mark.slow
@pytest.mark.parametrize("sample", ["romanian_deadlift_spec.json", "single_leg_romanian_deadlift_spec.json"])
def test_real_manim_render_of_the_samples(tmp_path, sample):
    # Arrange
    spec = AnimationSpec.model_validate(
        json.loads((Path("samples") / sample).read_text(encoding="utf-8"))
    )
    video = tmp_path / "out.mp4"
    image = tmp_path / "out.png"

    # Act
    render_animation(spec, "Romanian Deadlift (Barbell)", "barbell", ["hamstrings", "glutes"], video)
    extract_still(video, image, sk.peak_activation_time(spec) + 0.3)

    # Assert
    assert video.stat().st_size > 10_000
    assert image.stat().st_size > 1_000
