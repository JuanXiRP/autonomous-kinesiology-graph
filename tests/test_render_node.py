import json
from pathlib import Path

from src.domain.animation import AnimationSpec
from src.nodes import render_node
from tests.conftest import make_exercise

GOLDEN = Path("samples/specs/romanian_deadlift_barbell.json").resolve()


def golden_spec() -> AnimationSpec:
    return AnimationSpec.model_validate(json.loads(GOLDEN.read_text(encoding="utf-8")))


def test_success_sets_video_and_still(monkeypatch, tmp_path):
    # Arrange
    monkeypatch.chdir(tmp_path)

    def fake_render(spec, title, equipment, target, video_path):
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        Path(video_path).write_bytes(b"mp4")
        return video_path

    def fake_still(video_path, image_path, at_seconds):
        Path(image_path).parent.mkdir(parents=True, exist_ok=True)
        Path(image_path).write_bytes(b"png")

    monkeypatch.setattr(render_node, "render_animation", fake_render)
    monkeypatch.setattr(render_node, "extract_still", fake_still)

    # Act
    result = render_node.render_exercise_animation(
        {"structured_exercise": make_exercise(), "animation_spec": golden_spec()}
    )

    # Assert
    exercise = result["structured_exercise"]
    assert exercise.animation_video.endswith("romanian_deadlift_barbell_generative.mp4")
    assert exercise.images[0].endswith("romanian_deadlift_barbell.png")
    assert result["validation_errors"] == []


def test_render_failure_is_reported_and_leaves_media_empty(monkeypatch, tmp_path):
    # Arrange
    monkeypatch.chdir(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("manim crashed")

    monkeypatch.setattr(render_node, "render_animation", boom)
    exercise = make_exercise()

    # Act
    result = render_node.render_exercise_animation(
        {"structured_exercise": exercise, "animation_spec": golden_spec()}
    )

    # Assert
    assert "manim crashed" in result["validation_errors"][0]
    assert exercise.animation_video is None
    assert exercise.images == []


def test_still_failure_keeps_the_video(monkeypatch, tmp_path):
    # Arrange
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(render_node, "render_animation", lambda *a, **k: a[-1])

    def boom(*args, **kwargs):
        raise RuntimeError("no frame")

    monkeypatch.setattr(render_node, "extract_still", boom)

    # Act
    result = render_node.render_exercise_animation(
        {"structured_exercise": make_exercise(), "animation_spec": golden_spec()}
    )

    # Assert
    assert result["structured_exercise"].animation_video is not None
    assert result["structured_exercise"].images == []
    assert "no frame" in result["validation_errors"][0]


def test_missing_spec_reports_error():
    # Arrange / Act
    result = render_node.render_exercise_animation({"structured_exercise": make_exercise(), "animation_spec": None})

    # Assert
    assert "No animation spec" in result["validation_errors"][0]


def test_stale_output_is_deleted_before_rendering(monkeypatch, tmp_path):
    # Arrange
    monkeypatch.chdir(tmp_path)
    stale = tmp_path / "media" / "videos" / "romanian_deadlift_barbell_generative.mp4"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"")

    def boom(*args, **kwargs):
        raise RuntimeError("manim crashed")

    monkeypatch.setattr(render_node, "render_animation", boom)

    # Act
    render_node.render_exercise_animation({"structured_exercise": make_exercise(), "animation_spec": golden_spec()})

    # Assert
    assert not stale.exists()


def test_spec_is_saved_next_to_the_media(monkeypatch, tmp_path):
    # Arrange
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(render_node, "render_animation", lambda *a: a[-1])
    monkeypatch.setattr(render_node, "extract_still", lambda *a: None)

    # Act
    render_node.render_exercise_animation({"structured_exercise": make_exercise(), "animation_spec": golden_spec()})

    # Assert
    saved = tmp_path / "media" / "specs" / "romanian_deadlift_barbell.json"
    assert AnimationSpec.model_validate_json(saved.read_text(encoding="utf-8")) == golden_spec()


def test_fallback_spec_is_rendered_when_no_spec_was_approved(monkeypatch, tmp_path):
    # Arrange
    monkeypatch.chdir(tmp_path)
    rendered = []
    monkeypatch.setattr(render_node, "render_animation", lambda spec, *a: rendered.append(spec) or a[-1])
    monkeypatch.setattr(render_node, "extract_still", lambda *a: None)

    # Act
    render_node.render_exercise_animation(
        {"structured_exercise": make_exercise(), "animation_spec": None, "fallback_spec": golden_spec()})

    # Assert
    assert rendered == [golden_spec()]
