import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from src.domain.animation import AnimationSpec
from src.rendering import checks
from src.rendering import kinematics as kin
from src.rendering import scenery as scn
from src.rendering import skeleton as sk

SAMPLES = Path("samples/specs")


def data(name: str) -> dict:
    return json.loads((SAMPLES / name).read_text(encoding="utf-8"))


def load(name: str) -> AnimationSpec:
    return AnimationSpec.model_validate(data(name))


def test_front_specs_get_the_front_solver():
    # Arrange / Act / Assert
    assert isinstance(kin.Solver(load("lateral_raise_dumbbell.json")), kin.FrontSolver)
    assert type(kin.Solver(load("bench_press_barbell.json"))) is kin.Solver


def test_mirrored_far_arm_is_the_reflection_of_the_near_arm():
    # Arrange
    solver = kin.Solver(load("lateral_raise_dumbbell.json"))

    # Act
    points = solver.keyframe_points(2)
    midline = points["hip"][0]

    # Assert: same height, opposite sides of the body midline
    for joint in ("elbow", "wrist"):
        near, far = points[f"near_{joint}"], points[f"far_{joint}"]
        assert np.isclose(near[1], far[1])
        assert np.isclose(near[0] - midline, midline - far[0])


def test_lateral_raise_reaches_shoulder_height():
    # Arrange
    solver = kin.Solver(load("lateral_raise_dumbbell.json"))

    # Act
    abduction = checks.front_flexions(solver.keyframe_points(2))["shoulder"]

    # Assert
    assert 80 <= abduction <= 100


def test_stance_leg_is_the_one_that_does_not_move():
    # Arrange
    solver = kin.Solver(load("hip_abduction_cable.json"))
    total = sk.total_duration(solver.spec)

    # Act
    frames = [solver.at(total * i / 30) for i in range(31)]

    # Assert: the near leg abducts, the far foot stays planted
    assert solver.stance == "far"
    assert len({tuple(np.round(p["far_ankle"], 6)) for p in frames}) == 1
    assert max(p["near_ankle"][1] for p in frames) > solver.floor_y + 0.5


def test_bilateral_front_movements_load_both_hands():
    # Arrange
    solver = kin.Solver(load("cable_crossover_cable.json"))

    # Act
    loads = solver.loads(solver.keyframe_points(0))

    # Assert
    assert [side for side, _ in loads] == ["near", "far"]


def test_crossover_cables_come_from_both_columns():
    # Arrange
    spec = load("cable_crossover_cable.json")
    solver = kin.Solver(spec)
    scenery = scn.build_scenery(spec, solver, "cable")
    points = solver.keyframe_points(0)

    # Act
    cables = scn.moving_parts(spec, scenery, solver.load(points), solver.loads(points))

    # Assert: each hand pulls from the column on its own side
    assert len(cables) == 2
    near_cable, far_cable = cables
    assert near_cable.a[0] > points["hip"][0] > far_cable.a[0]


def test_abduction_beyond_range_is_flagged():
    # Arrange: leg swung up to the side far past the hip's range
    spec = data("hip_abduction_cable.json")
    spec["keyframes"][2]["near_leg"] = {"proximal": 355, "distal": 355}

    # Act
    problems = checks.check_spec_geometry(AnimationSpec.model_validate(spec), "cable")

    # Assert
    assert any("near hip abduction" in p and "beyond the human range" in p for p in problems)


def test_arm_crossing_far_over_the_midline_is_flagged():
    # Arrange
    spec = data("lateral_raise_dumbbell.json")
    spec["keyframes"][2]["near_arm"] = {"proximal": 215, "distal": 215}

    # Act
    problems = checks.check_spec_geometry(AnimationSpec.model_validate(spec), "dumbbell")

    # Assert
    assert any("crosses the body midline" in p for p in problems)


def test_front_view_rejects_lying_postures():
    # Arrange
    spec = data("lateral_raise_dumbbell.json")
    spec["setup"] = {"posture": "supine", "support": "flat_bench"}

    # Act / Assert
    with pytest.raises(ValidationError, match="view 'front'"):
        AnimationSpec.model_validate(spec)


@pytest.mark.parametrize("rig", ["machine_linear", "smith"])
def test_front_view_rejects_rails(rig):
    # Arrange
    spec = data("lateral_raise_dumbbell.json")
    spec["setup"]["rig"] = rig

    # Act / Assert
    with pytest.raises(ValidationError, match="lever machines only"):
        AnimationSpec.model_validate(spec)


def test_bend_hint_chooses_the_elbow_side():
    # Arrange: hand straight below the shoulder, half bent
    spec = data("lateral_raise_dumbbell.json")
    for hint in ("out", "in"):
        for keyframe in spec["keyframes"]:
            keyframe["near_arm"] = {"target": {"angle_deg": 270, "reach": 0.6}, "bend": hint}
        solver = kin.Solver(AnimationSpec.model_validate(spec))

        # Act
        points = solver.keyframe_points(0)

        # Assert
        outward = points["near_elbow"][0] - points["near_shoulder"][0]
        assert (outward > 0) == (hint == "out")


def test_lateral_raise_machine_levers_turn_about_the_shoulders():
    # Arrange
    spec = load("lateral_raise_machine.json")
    solver = kin.Solver(spec)

    # Act
    scenery = scn.build_scenery(spec, solver, "machine")
    points = solver.keyframe_points(0)

    # Assert: the pad (elbow) circles the shoulder joint, so the fitted pivot sits on it; far is the mirror
    assert np.linalg.norm(scenery.pivots["near"] - points["near_shoulder"]) < 0.3
    assert np.linalg.norm(scenery.pivots["far"] - points["far_shoulder"]) < 0.3


def test_machine_pads_load_both_elbows_and_get_one_lever_each():
    # Arrange
    spec = load("lateral_raise_machine.json")
    solver = kin.Solver(spec)
    scenery = scn.build_scenery(spec, solver, "machine")
    points = solver.keyframe_points(2)

    # Act
    loads = solver.loads(points)
    levers = scn.moving_parts(spec, scenery, solver.load(points), loads)

    # Assert
    assert [(side, tuple(p)) for side, p in loads] == [("near", tuple(points["near_elbow"])), ("far", tuple(points["far_elbow"]))]
    assert [part.kind for part in levers] == ["lever", "lever"]


def test_knee_pads_load_both_knees():
    # Arrange
    solver = kin.Solver(load("hip_abduction_machine.json"))

    # Act
    points = solver.keyframe_points(2)

    # Assert
    assert [tuple(p) for _, p in solver.loads(points)] == [tuple(points["near_knee"]), tuple(points["far_knee"])]


def test_seated_front_thigh_is_foreshortened_but_abduction_is_measured_in_full():
    # Arrange
    spec = load("hip_abduction_machine.json")
    solver = kin.Solver(spec)

    # Act
    together, spread = solver.keyframe_points(0), solver.keyframe_points(2)

    # Assert: knees together sit just below the hips; spread, the real 48° abduction is recovered
    drop = together["near_hip"][1] - together["near_knee"][1]
    assert 0 < drop < 0.3 * sk.BONES["thigh"]
    assert abs(checks.front_flexions(spread, seated=True)["hip"] - 48) < 1.0
    assert all(min(p["near_heel"][1], p["near_toe"][1]) >= solver.floor_y - 0.05 for p in (together, spread))
