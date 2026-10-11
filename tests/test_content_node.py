from langchain_core.runnables import RunnableLambda

from src.nodes import content_node
from tests.conftest import make_exercise


class FakeLLM:
    def __init__(self, fn):
        self._fn = fn

    def with_structured_output(self, schema):
        return RunnableLambda(self._fn)


def _install_fake(monkeypatch, fn):
    monkeypatch.setattr(content_node, "ChatGoogleGenerativeAI", lambda **kwargs: FakeLLM(fn))


def test_invented_media_fields_are_cleared(monkeypatch):
    # Arrange
    _install_fake(
        monkeypatch,
        lambda _: make_exercise(images=["invented.jpg"], animation_video="invented.mp4"),
    )

    # Act
    result = content_node.generate_exercise_content(
        {"raw_exercise_name": "Romanian Deadlift", "generation_attempts": 0}
    )

    # Assert
    assert result["structured_exercise"].images == []
    assert result["structured_exercise"].animation_video is None
    assert result["generation_attempts"] == 1


def test_llm_failure_is_reported_not_raised(monkeypatch):
    # Arrange
    def boom(_):
        raise RuntimeError("llm down")

    _install_fake(monkeypatch, boom)

    # Act
    result = content_node.generate_exercise_content(
        {"raw_exercise_name": "Romanian Deadlift", "generation_attempts": 1}
    )

    # Assert
    assert len(result["validation_errors"]) == 1
    assert "llm down" in result["validation_errors"][0]
    assert result["generation_attempts"] == 2


def test_chosen_variant_pins_name_equipment_and_slug(monkeypatch):
    # Arrange
    from src.domain.models import EquipmentType
    from src.resolution.variants import ExerciseVariant

    chosen = ExerciseVariant(name="Single-Leg Romanian Deadlift", equipment=EquipmentType.DUMBBELL,
                             label_es="Peso muerto rumano unilateral (mancuerna)", note_es="-")
    seen = {}

    def answer(prompt_value):
        seen["prompt"] = prompt_value.to_string()
        return make_exercise(name="Romanian Deadlift", equipment="barbell")

    _install_fake(monkeypatch, answer)

    # Act
    result = content_node.generate_exercise_content(
        {"raw_exercise_name": chosen.full_name, "variant": chosen, "generation_attempts": 0})

    # Assert
    exercise = result["structured_exercise"]
    assert exercise.slug == "single-leg_romanian_deadlift_dumbbell"
    assert exercise.equipment == EquipmentType.DUMBBELL
    assert "Single-Leg Romanian Deadlift" in seen["prompt"]
