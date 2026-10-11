import json
import math
from pathlib import Path

import numpy as np

from src.domain.animation import AnchorKind, AnimationSpec
from src.rendering import skeleton as sk

GOLDEN = Path("samples/romanian_deadlift_spec.json")


def golden_spec() -> AnimationSpec:
    return AnimationSpec.model_validate(json.loads(GOLDEN.read_text(encoding="utf-8")))


def test_bone_lengths_are_constant_across_the_whole_timeline():
    # Arrange
    spec = golden_spec()
    total = sk.total_duration(spec)

    # Act
    lengths = set()
    for i in range(0, 101):
        points = sk.solve(sk.frame_at(spec, total * i / 100).pose, spec.anchor)
        lengths.add((
            round(float(np.linalg.norm(points["knee"] - points["ankle"])), 6),
            round(float(np.linalg.norm(points["hip"] - points["knee"])), 6),
            round(float(np.linalg.norm(points["shoulder"] - points["hip"])), 6),
        ))

    # Assert
    assert len(lengths) == 1


def test_feet_stay_on_the_floor_and_planted():
    # Arrange
    spec = golden_spec()
    total = sk.total_duration(spec)

    # Act
    frames = [sk.solve(sk.frame_at(spec, total * i / 50).pose, spec.anchor) for i in range(51)]

    # Assert
    assert all(abs(min(p["heel"][1], p["toe"][1]) - sk.FLOOR_Y) < 1e-6 for p in frames)
    assert all(abs(p["ankle"][0] - sk.FEET_ANCHOR_X) < 1e-6 for p in frames)


def test_golden_spec_has_no_geometry_problems():
    # Arrange / Act / Assert
    assert sk.check_spec_geometry(golden_spec()) == []


def test_backwards_knee_is_flagged():
    # Arrange: shank leans backwards and the thigh forwards -> the knee points to the left
    pose = sk.Pose(shank=110, thigh=70, torso=90, upper_arm=270, forearm=270)
    points = sk.solve(pose)

    # Act
    problems = sk.check_pose(points, "test")

    # Assert
    assert any("knee bends backwards" in problem for problem in problems)


def test_impossible_flexion_is_flagged():
    # Arrange: thigh folded flat against the shank
    pose = sk.Pose(shank=90, thigh=-80, torso=90, upper_arm=270, forearm=270)

    # Act
    problems = sk.check_pose(sk.solve(pose), "test")

    # Assert
    assert any("beyond the human range" in problem for problem in problems)


def test_timeline_loops_back_to_the_first_pose():
    # Arrange
    spec = golden_spec()
    rep = sk.build_timeline(spec).rep_duration

    # Act
    start = sk.frame_at(spec, 0.0).pose
    next_rep_start = sk.frame_at(spec, rep + 0.01).pose

    # Assert
    assert math.isclose(start.torso, next_rep_start.torso, abs_tol=1.0)


def test_hanging_anchor_keeps_the_wrist_fixed():
    # Arrange
    spec = golden_spec().model_copy(update={"anchor": AnchorKind.HANDS})
    hand_y = sk.hand_anchor_y(spec)

    # Act
    wrists = [sk.solve(sk.frame_at(spec, t).pose, spec.anchor, hand_y)["wrist"] for t in (0.0, 2.0, 4.0)]

    # Assert
    assert all(np.allclose(w, wrists[0]) for w in wrists)


def test_peak_activation_is_the_bottom_of_the_romanian_deadlift():
    # Arrange
    spec = golden_spec()
    timeline = sk.build_timeline(spec)

    # Act
    peak = sk.peak_activation_time(spec)

    # Assert
    assert timeline.arrive[2] <= peak <= timeline.leave[2]


def test_golden_load_stays_over_midfoot():
    # Arrange
    spec = golden_spec()

    # Act
    offsets = [abs(sk.solve(sk.Pose.from_keyframe(kf), spec.anchor)["wrist"][0]
                   - sk.midfoot_x(sk.solve(sk.Pose.from_keyframe(kf), spec.anchor)))
               for kf in spec.keyframes]

    # Assert
    assert max(offsets) <= sk.MIDFOOT_TOLERANCE


def test_floating_bar_is_flagged_when_balance_is_required():
    # Arrange: torso leaning forward with vertical arms puts the bar far in front of the feet
    data = golden_spec().model_dump(mode="json")
    for keyframe in data["keyframes"][1:-1]:
        keyframe.update({"thigh": 100, "torso": 30, "upper_arm": 270, "forearm": 270})
    spec = AnimationSpec.model_validate(data)

    # Act
    problems = sk.check_spec_geometry(spec)

    # Assert
    assert any("in front of mid-foot" in problem for problem in problems)


def test_balance_rule_is_skipped_when_not_required():
    # Arrange
    data = golden_spec().model_dump(mode="json")
    data["balance_over_midfoot"] = False
    for keyframe in data["keyframes"][1:-1]:
        keyframe.update({"thigh": 100, "torso": 30, "upper_arm": 270, "forearm": 270})

    # Act
    problems = sk.check_spec_geometry(AnimationSpec.model_validate(data))

    # Assert
    assert not any("mid-foot" in problem for problem in problems)


SINGLE_LEG = Path("samples/single_leg_romanian_deadlift_spec.json")


def single_leg_spec() -> AnimationSpec:
    return AnimationSpec.model_validate(json.loads(SINGLE_LEG.read_text(encoding="utf-8")))


def test_single_leg_sample_has_no_geometry_problems():
    # Arrange / Act / Assert
    assert single_leg_spec().has_free_leg
    assert sk.check_spec_geometry(single_leg_spec()) == []


def test_free_leg_does_not_change_grounding():
    # Arrange
    with_free = sk.Pose(90, 90, 90, 270, 270, free_thigh=20, free_shank=20)
    without = sk.Pose(90, 90, 90, 270, 270)

    # Act
    a, b = sk.solve(with_free), sk.solve(without)

    # Assert
    assert np.allclose(a["ankle"], b["ankle"]) and np.allclose(a["hip"], b["hip"])


def test_free_leg_bone_lengths_are_constant():
    # Arrange
    spec = single_leg_spec()
    total = sk.total_duration(spec)

    # Act
    lengths = {
        (round(float(np.linalg.norm(p["hip"] - p["free_knee"])), 6),
         round(float(np.linalg.norm(p["free_knee"] - p["free_ankle"])), 6))
        for p in (sk.solve(sk.frame_at(spec, total * i / 40).pose, spec.anchor) for i in range(41))
    }

    # Assert
    assert len(lengths) == 1


def test_free_foot_through_the_floor_is_flagged():
    # Arrange: free leg pointing straight down and forward, below the stance foot
    points = sk.solve(sk.Pose(80, 110, 60, 270, 270, free_thigh=100, free_shank=95))

    # Act
    problems = sk.check_free_leg_clearance(points, "test")

    # Assert
    assert any("through the floor" in problem for problem in problems)
