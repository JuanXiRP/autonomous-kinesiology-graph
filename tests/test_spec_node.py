import json
from pathlib import Path

from langchain_core.runnables import RunnableLambda

from src.domain.animation import AnimationSpec
from src.nodes import spec_node
from tests.conftest import make_exercise

GOLDEN = Path("samples/romanian_deadlift_spec.json")


class FakeLLM:
    def __init__(self, fn):
        self._fn = fn

    def with_structured_output(self, schema):
        return RunnableLambda(self._fn)


def install_fake(monkeypatch, fn):
    monkeypatch.setattr(spec_node, "ChatGoogleGenerativeAI", lambda **kwargs: FakeLLM(fn))


def golden_spec() -> AnimationSpec:
    return AnimationSpec.model_validate(json.loads(GOLDEN.read_text(encoding="utf-8")))


def test_valid_spec_is_stored_in_the_state(monkeypatch):
    # Arrange
    install_fake(monkeypatch, lambda _: golden_spec())

    # Act
    result = spec_node.generate_animation_spec({"structured_exercise": make_exercise(), "spec_attempts": 0})

    # Assert
    assert isinstance(result["animation_spec"], AnimationSpec)
    assert result["spec_attempts"] == 1


def test_geometry_problem_is_reported_not_raised(monkeypatch):
    # Arrange
    bad = golden_spec()
    for keyframe in bad.keyframes:
        keyframe.shank, keyframe.thigh = 110, 70  # knee bends backwards in every pose
    install_fake(monkeypatch, lambda _: bad)

    # Act
    result = spec_node.generate_animation_spec({"structured_exercise": make_exercise(), "spec_attempts": 0})

    # Assert
    assert "animation_spec" not in result
    assert "knee bends backwards" in result["validation_errors"][0]


def test_llm_failure_is_reported_not_raised(monkeypatch):
    # Arrange
    def boom(_):
        raise RuntimeError("llm down")

    install_fake(monkeypatch, boom)

    # Act
    result = spec_node.generate_animation_spec({"structured_exercise": make_exercise(), "spec_attempts": 1})

    # Assert
    assert result["spec_attempts"] == 2
    assert "llm down" in result["validation_errors"][0]


def test_previous_errors_are_sent_back_to_the_llm(monkeypatch):
    # Arrange
    seen = {}

    def capture(messages):
        seen["human"] = messages[1].content
        return golden_spec()

    install_fake(monkeypatch, capture)
    state = {
        "structured_exercise": make_exercise(),
        "spec_attempts": 1,
        "validation_errors": ["Spec generation failed on attempt 1: knee bends backwards"],
    }

    # Act
    spec_node.generate_animation_spec(state)

    # Assert
    assert "knee bends backwards" in seen["human"]


def test_missing_exercise_reports_error():
    # Arrange / Act
    result = spec_node.generate_animation_spec({"structured_exercise": None})

    # Assert
    assert "No exercise data" in result["validation_errors"][0]
