import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.domain.animation import AnimationSpec

GOLDEN = Path("samples/romanian_deadlift_spec.json")


def golden_data() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_golden_spec_is_valid():
    # Arrange / Act
    spec = AnimationSpec.model_validate(golden_data())

    # Assert
    assert len(spec.keyframes) == 5
    assert len(spec.steps) == 4


def test_loop_must_close():
    # Arrange
    data = golden_data()
    data["keyframes"][-1]["torso"] = 40

    # Act / Assert
    with pytest.raises(ValidationError, match="last keyframe"):
        AnimationSpec.model_validate(data)


def test_step_description_too_long_is_rejected():
    # Arrange
    data = golden_data()
    data["steps"][0]["description"] = "x" * 91

    # Act / Assert
    with pytest.raises(ValidationError, match="90 characters"):
        AnimationSpec.model_validate(data)


def test_active_step_must_exist():
    # Arrange
    data = golden_data()
    data["keyframes"][1]["active_step"] = 9

    # Act / Assert
    with pytest.raises(ValidationError, match="active_step"):
        AnimationSpec.model_validate(data)


def test_muscle_level_must_be_between_zero_and_one():
    # Arrange
    data = golden_data()
    data["keyframes"][1]["muscle_activation"][0]["level"] = 1.7

    # Act / Assert
    with pytest.raises(ValidationError, match="between 0 and 1"):
        AnimationSpec.model_validate(data)


def test_cable_anchor_needs_a_height():
    # Arrange
    data = golden_data()
    data["props"] = [{"kind": "cable_anchor", "x": 2.0, "y": 0.0}]

    # Act / Assert
    with pytest.raises(ValidationError, match="cable_anchor"):
        AnimationSpec.model_validate(data)


def test_balance_flag_defaults_to_false():
    # Arrange
    data = golden_data()
    del data["balance_over_midfoot"]

    # Act / Assert
    assert AnimationSpec.model_validate(data).balance_over_midfoot is False


def test_free_leg_angles_must_come_in_pairs():
    # Arrange
    data = golden_data()
    for keyframe in data["keyframes"]:
        keyframe["free_thigh"] = 80

    # Act / Assert
    with pytest.raises(ValidationError, match="set together"):
        AnimationSpec.model_validate(data)


def test_free_leg_must_be_in_every_keyframe():
    # Arrange
    data = golden_data()
    data["keyframes"][0].update({"free_thigh": 80, "free_shank": 70})

    # Act / Assert
    with pytest.raises(ValidationError, match="every keyframe or in none"):
        AnimationSpec.model_validate(data)


def test_loop_closure_includes_the_free_leg():
    # Arrange
    data = golden_data()
    for keyframe in data["keyframes"]:
        keyframe.update({"free_thigh": 80, "free_shank": 70})
    data["keyframes"][-1]["free_thigh"] = 40

    # Act / Assert
    with pytest.raises(ValidationError, match="free_thigh differs"):
        AnimationSpec.model_validate(data)
