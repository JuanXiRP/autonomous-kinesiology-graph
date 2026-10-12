import json
from pathlib import Path

from langchain_core.runnables import RunnableLambda

from src.domain.animation import AnimationSpec
from src.nodes import critic_node
from src.nodes.critic_node import VisualReview
from tests.conftest import make_exercise

GOLDEN = Path("samples/specs/romanian_deadlift_barbell.json")


def golden_spec() -> AnimationSpec:
    return AnimationSpec.model_validate(json.loads(GOLDEN.read_text(encoding="utf-8")))


class FakeLLM:
    def __init__(self, fn):
        self._fn = fn

    def with_structured_output(self, schema):
        return RunnableLambda(self._fn)


def install(monkeypatch, tmp_path, review_fn):
    image = tmp_path / "storyboard.png"
    image.write_bytes(b"png")
    monkeypatch.setattr(critic_node, "render_storyboard", lambda *args: image)
    monkeypatch.setattr(critic_node, "ChatGoogleGenerativeAI", lambda **kwargs: FakeLLM(review_fn))


def state(attempts: int = 1) -> dict:
    return {"structured_exercise": make_exercise(), "animation_spec": golden_spec(), "spec_attempts": attempts}


def test_approved_spec_is_kept(monkeypatch, tmp_path):
    # Arrange
    install(monkeypatch, tmp_path, lambda _: VisualReview(approved=True, problems=[]))

    # Act
    result = critic_node.review_animation_spec(state())

    # Assert
    assert result["animation_spec"] == golden_spec()
    assert "validation_errors" not in result


def test_rejection_clears_the_spec_and_reports_the_problems(monkeypatch, tmp_path):
    # Arrange
    install(monkeypatch, tmp_path, lambda _: VisualReview(approved=False, problems=["the bar floats above the hands"]))

    # Act
    result = critic_node.review_animation_spec(state(attempts=2))

    # Assert
    assert result["animation_spec"] is None
    assert result["fallback_spec"] == golden_spec()
    assert result["validation_errors"] == ["Visual review rejected (attempt 2): the bar floats above the hands"]


def test_the_storyboard_image_is_sent_to_the_model(monkeypatch, tmp_path):
    # Arrange
    seen = {}

    def capture(messages):
        seen["content"] = messages[1].content
        return VisualReview(approved=True)

    install(monkeypatch, tmp_path, capture)

    # Act
    critic_node.review_animation_spec(state())

    # Assert
    kinds = [part["type"] for part in seen["content"]]
    assert kinds == ["text", "image_url"]
    assert seen["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "Romanian Deadlift" in seen["content"][0]["text"]


def test_api_failure_approves_and_never_raises(monkeypatch, tmp_path):
    # Arrange
    def boom(_):
        raise RuntimeError("quota exceeded")

    install(monkeypatch, tmp_path, boom)

    # Act
    result = critic_node.review_animation_spec(state())

    # Assert
    assert "animation_spec" not in result
    assert "quota exceeded" in result["validation_errors"][0]


def test_storyboard_failure_approves_and_never_raises(monkeypatch, tmp_path):
    # Arrange
    def broken(*args):
        raise RuntimeError("manim crashed")

    monkeypatch.setattr(critic_node, "render_storyboard", broken)

    # Act
    result = critic_node.review_animation_spec(state())

    # Assert
    assert "manim crashed" in result["validation_errors"][0]
