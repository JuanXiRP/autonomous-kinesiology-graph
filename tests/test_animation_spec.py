import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.domain.animation import AnimationSpec, Setup

GOLDEN = Path("samples/specs/romanian_deadlift_barbell.json")
SINGLE_LEG = Path("samples/specs/single_leg_romanian_deadlift_dumbbell.json")


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


def test_loop_closure_includes_the_limbs():
    # Arrange
    data = golden_data()
    data["keyframes"][-1]["near_arm"] = {"proximal": 200, "distal": 270}

    # Act / Assert
    with pytest.raises(ValidationError, match="near_arm differs"):
        AnimationSpec.model_validate(data)


def test_loop_closure_compares_targets():
    # Arrange
    data = golden_data()
    data["keyframes"][0]["near_arm"] = {"target": {"angle_deg": 270, "reach": 1.0}}
    data["keyframes"][-1]["near_arm"] = {"target": {"angle_deg": 270, "reach": 0.8}}

    # Act / Assert
    with pytest.raises(ValidationError, match="near_arm differs"):
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


def test_balance_flag_defaults_to_false():
    # Arrange
    data = golden_data()
    del data["balance_over_midfoot"]

    # Act / Assert
    assert AnimationSpec.model_validate(data).balance_over_midfoot is False


def test_limb_needs_exactly_one_mode():
    # Arrange
    data = golden_data()
    data["keyframes"][1]["near_arm"] = {"proximal": 260, "distal": 260, "target": {"angle_deg": 270, "reach": 0.9}}

    # Act / Assert
    with pytest.raises(ValidationError, match="either proximal"):
        AnimationSpec.model_validate(data)


def test_limb_angles_come_in_pairs():
    # Arrange
    data = golden_data()
    data["keyframes"][1]["near_arm"] = {"proximal": 260}

    # Act / Assert
    with pytest.raises(ValidationError, match="needs both proximal and distal"):
        AnimationSpec.model_validate(data)


def test_reach_must_be_in_range():
    # Arrange
    data = golden_data()
    data["keyframes"][1]["near_arm"] = {"target": {"angle_deg": 270, "reach": 1.4}}

    # Act / Assert
    with pytest.raises(ValidationError, match="reach must be between"):
        AnimationSpec.model_validate(data)


def test_far_leg_must_be_in_every_keyframe_or_none():
    # Arrange
    data = golden_data()
    data["keyframes"][0]["far_leg"] = {"proximal": 250, "distal": 250}

    # Act / Assert
    with pytest.raises(ValidationError, match="every keyframe or in none"):
        AnimationSpec.model_validate(data)


def test_single_leg_sample_has_an_independent_far_leg():
    # Arrange / Act
    spec = AnimationSpec.model_validate(json.loads(SINGLE_LEG.read_text(encoding="utf-8")))

    # Assert
    assert spec.has_independent("far_leg")
    assert not spec.has_independent("far_arm")


def test_support_must_match_the_posture():
    # Arrange / Act / Assert
    with pytest.raises(ValidationError, match="needs posture"):
        Setup(posture="standing", support="incline_bench", support_angle_deg=30)


def test_incline_angle_is_range_checked():
    # Arrange / Act / Assert
    with pytest.raises(ValidationError, match="support_angle_deg for incline_bench"):
        Setup(posture="supine", support="incline_bench", support_angle_deg=5)


def test_flat_bench_angle_is_forced_to_zero():
    # Arrange / Act
    setup = Setup(posture="supine", support="flat_bench", support_angle_deg=20)

    # Assert
    assert setup.support_angle_deg == 0


def test_cable_rig_needs_a_pulley():
    # Arrange / Act / Assert
    with pytest.raises(ValidationError, match="cable pulley"):
        Setup(posture="standing", rig="cable")


def test_hanging_needs_a_bar():
    # Arrange / Act / Assert
    with pytest.raises(ValidationError, match="hanging"):
        Setup(posture="hanging", support="none")
