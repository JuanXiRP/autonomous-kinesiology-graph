import json
import math
from pathlib import Path

import numpy as np

from src.domain.animation import AnimationSpec
from src.rendering import skeleton as sk

GOLDEN = Path("samples/specs/romanian_deadlift_barbell.json")
BENCH = Path("samples/specs/bench_press_barbell.json")


def load(path: Path) -> AnimationSpec:
    return AnimationSpec.model_validate(json.loads(path.read_text(encoding="utf-8")))


def test_timeline_loops_back_to_the_first_pose():
    # Arrange
    spec = load(GOLDEN)
    rep = sk.build_timeline(spec).rep_duration

    # Act
    start = sk.frame_at(spec, 0.0).pose
    next_rep_start = sk.frame_at(spec, rep + 0.01).pose

    # Assert
    assert math.isclose(start.torso, next_rep_start.torso, abs_tol=1.0)


def test_peak_activation_is_the_bottom_of_the_romanian_deadlift():
    # Arrange
    spec = load(GOLDEN)
    timeline = sk.build_timeline(spec)

    # Act
    peak = sk.peak_activation_time(spec)

    # Assert
    assert timeline.arrive[2] <= peak <= timeline.leave[2]


def test_angle_limbs_interpolate_their_angles():
    # Arrange
    a = sk.LimbState(proximal=270, distal=270)
    b = sk.LimbState(proximal=250, distal=230)

    # Act
    mid = a.lerp(b, 0.5, "near_arm")

    # Assert
    assert (mid.proximal, mid.distal) == (260, 250)


def test_target_limbs_interpolate_in_a_straight_line():
    # Arrange
    a = sk.LimbState(vector=np.array([0.0, 1.0]))
    b = sk.LimbState(vector=np.array([1.0, 0.0]))

    # Act
    mid = a.lerp(b, 0.5, "near_arm")

    # Assert
    assert np.allclose(mid.vector, [0.5, 0.5])


def test_mixed_modes_interpolate_the_hand_position():
    # Arrange: straight arm hanging down vs a target in front
    a = sk.LimbState(proximal=270, distal=270)
    b = sk.LimbState(vector=np.array([1.0, 0.0]))

    # Act
    mid = a.lerp(b, 0.5, "near_arm")

    # Assert
    hanging = np.array([0.0, -sk.limb_length("near_arm")])
    assert np.allclose(mid.vector, (hanging + b.vector) / 2)


def test_mirrored_far_limb_returns_the_near_one():
    # Arrange
    spec = load(BENCH)

    # Act
    pose = sk.Pose.from_keyframe(spec.keyframes[0])

    # Assert
    assert pose.far_arm is None
    assert pose.limb("far_arm") is pose.near_arm


def test_sample_times_include_every_keyframe():
    # Arrange
    spec = load(GOLDEN)

    # Act
    times = sk.sample_times(spec, 10)

    # Assert
    assert set(sk.build_timeline(spec).arrive[:-1]) <= set(times)
