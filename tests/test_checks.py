import json
from pathlib import Path

from src.domain.animation import AnimationSpec
from src.rendering import checks
from src.rendering import kinematics as kin
from src.rendering import skeleton as sk

SAMPLES = Path("samples/specs")


def data(name: str) -> dict:
    return json.loads((SAMPLES / name).read_text(encoding="utf-8"))


def problems_for(spec_data: dict, equipment: str = "barbell") -> list[str]:
    return checks.check_spec_geometry(AnimationSpec.model_validate(spec_data), equipment)


def test_backwards_knee_is_flagged():
    # Arrange: thigh pointing down-back and the shank down-forward
    spec = data("romanian_deadlift_barbell.json")
    for keyframe in spec["keyframes"]:
        keyframe["near_leg"] = {"proximal": 250, "distal": 290}

    # Act / Assert
    assert any("knee bends backwards" in p for p in problems_for(spec))


def test_impossible_elbow_flexion_is_flagged():
    # Arrange: forearm folded back onto the upper arm
    spec = data("romanian_deadlift_barbell.json")
    spec["keyframes"][2]["near_arm"] = {"proximal": 270, "distal": 270 + 175}

    # Act / Assert
    assert any("elbow flexion" in p and "beyond the human range" in p for p in problems_for(spec))



def test_hyperextended_shoulder_is_flagged():
    # Arrange: arm swung up behind the leaning torso
    spec = data("romanian_deadlift_barbell.json")
    spec["keyframes"][1]["near_arm"] = {"proximal": 120, "distal": 120}

    # Act / Assert
    assert any("arm goes too far behind the torso" in p for p in problems_for(spec))


def test_bar_through_the_chest_is_flagged():
    # Arrange: bench press bar pulled into the torso
    spec = data("bench_press_barbell.json")
    spec["keyframes"][2]["near_arm"] = {"target": {"angle_deg": 330, "reach": 0.32}}

    # Act
    found = problems_for(spec)

    # Assert
    assert any(p.startswith("keyframe 2: the load sinks") and "torso" in p for p in found)


def test_bar_through_the_head_is_flagged():
    # Arrange: bench press bar lowered behind and onto the head
    spec = data("bench_press_barbell.json")
    spec["keyframes"][2]["near_arm"] = {"target": {"angle_deg": 175, "reach": 0.4}}

    # Act / Assert
    assert any("through the head" in p for p in problems_for(spec))


def test_dumbbells_may_pass_beside_the_body():
    # Arrange: same pose as the bar through the chest, but dumbbells
    spec = data("bench_press_barbell.json")
    spec["keyframes"][2]["near_arm"] = {"target": {"angle_deg": 330, "reach": 0.32}}

    # Act / Assert
    assert not any("sinks" in p for p in problems_for(spec, "dumbbell"))


def test_torso_off_the_bench_is_flagged():
    # Arrange
    spec = data("bench_press_barbell.json")
    for keyframe in spec["keyframes"]:
        keyframe["torso"] = 130

    # Act / Assert
    assert any("leaves the flat bench" in p and "torso ≈ 180" in p for p in problems_for(spec))


def test_planted_foot_out_of_reach_is_flagged():
    # Arrange: split stance with the back leg already straight, then the hips travel forward
    spec = data("romanian_deadlift_barbell.json")
    spec["setup"]["fixed_contacts"] = ["far_foot"]
    spec["balance_over_midfoot"] = False
    for index, keyframe in enumerate(spec["keyframes"]):
        keyframe["far_leg"] = {"proximal": 235, "distal": 235}
        keyframe["near_leg"] = ({"proximal": 270, "distal": 270} if index in (0, 4)
                                else {"target": {"angle_deg": 240, "reach": 1.0}})

    # Act / Assert
    assert any("cannot stay planted" in p for p in problems_for(spec))


def test_floating_bar_is_flagged_when_balance_is_required():
    # Arrange: torso leaning forward with vertical arms puts the bar far in front of the feet
    spec = data("romanian_deadlift_barbell.json")
    for keyframe in spec["keyframes"][1:-1]:
        keyframe.update({"torso": 30, "near_arm": {"proximal": 270, "distal": 270},
                         "near_leg": {"proximal": 280, "distal": 270}})

    # Act / Assert
    assert any("in front of mid-foot" in p for p in problems_for(spec))


def test_balance_rule_is_skipped_when_not_required():
    # Arrange
    spec = data("romanian_deadlift_barbell.json")
    spec["balance_over_midfoot"] = False
    for keyframe in spec["keyframes"][1:-1]:
        keyframe.update({"torso": 30, "near_arm": {"proximal": 270, "distal": 270},
                         "near_leg": {"proximal": 280, "distal": 270}})

    # Act / Assert
    assert not any("mid-foot" in p for p in problems_for(spec))


def test_free_leg_through_the_floor_is_flagged():
    # Arrange: free leg pointing down and forward, below the stance foot
    spec = data("single_leg_romanian_deadlift_dumbbell.json")
    for keyframe in spec["keyframes"]:
        keyframe["far_leg"] = {"proximal": 280, "distal": 285}
        keyframe["near_leg"] = {"proximal": 300, "distal": 240}

    # Act / Assert
    assert any("through the floor" in p for p in problems_for(spec, "dumbbell"))


def test_smith_bar_must_move_vertically():
    # Arrange: bench press on a smith machine with a slanted path
    spec = data("bench_press_barbell.json")
    spec["setup"]["rig"] = "smith"
    spec["keyframes"][2]["near_arm"] = {"target": {"angle_deg": 30, "reach": 0.5}}

    # Act / Assert
    assert any("smith" in p for p in problems_for(spec))


def test_flexions_are_zero_standing_tall():
    # Arrange
    spec = AnimationSpec.model_validate(data("romanian_deadlift_barbell.json"))

    # Act
    bent = checks.flexions(kin.Solver(spec).solve(sk.Pose.from_keyframe(spec.keyframes[0])))

    # Assert
    assert all(abs(value) < 1e-6 for value in bent.values())


def test_collisions_are_reported_once_per_transition():
    # Arrange: bar sunk into the chest at the bottom of the bench press, reached and left again
    spec = data("bench_press_barbell.json")
    spec["keyframes"][2]["near_arm"] = {"target": {"angle_deg": 330, "reach": 0.32}}

    # Act
    sinks = [p for p in problems_for(spec) if "sinks" in p]

    # Assert: one line per transition/keyframe, not one per sampled frame
    labels = [p.split(":")[0] for p in sinks]
    assert len(labels) == len(set(labels)) >= 2
