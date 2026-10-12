import json
from pathlib import Path

import numpy as np
import pytest

from src.domain.animation import AnimationSpec, Contact
from src.rendering import kinematics as kin
from src.rendering import scenery as scn
from src.rendering import skeleton as sk
from src.rendering.checks import flexions

SAMPLES = Path("samples/specs")


def load(name: str) -> AnimationSpec:
    return AnimationSpec.model_validate(json.loads((SAMPLES / name).read_text(encoding="utf-8")))


def frames(spec: AnimationSpec, count: int = 40):
    solver = kin.Solver(spec)
    total = sk.total_duration(spec)
    return solver, [solver.at(total * i / count) for i in range(count + 1)]


def test_ik_reaches_a_reachable_target():
    # Arrange
    root, target = np.zeros(2), np.array([0.8, 0.6])

    # Act
    _, end = kin.ik(root, target, 0.9, 0.9, -1.0)

    # Assert
    assert np.allclose(end, target)


def test_ik_keeps_bone_lengths_when_the_target_is_out_of_reach():
    # Arrange
    root, target = np.zeros(2), np.array([5.0, 0.0])

    # Act
    middle, end = kin.ik(root, target, 0.9, 0.8, -1.0)

    # Assert
    assert np.isclose(np.linalg.norm(middle - root), 0.9)
    assert np.isclose(np.linalg.norm(end - middle), 0.8)


@pytest.mark.parametrize("angle", [0, 45, 90, 200, 300])
def test_ik_bends_the_elbow_and_knee_the_anatomical_way(angle):
    # Arrange
    pose = sk.Pose(torso=90,
                   near_arm=sk.LimbState(vector=0.5 * sk.limb_length("near_arm") * sk.direction(angle)),
                   near_leg=sk.LimbState(vector=0.6 * sk.limb_length("near_leg") * sk.direction(290)))
    spec = load("bench_press_barbell.json")

    # Act
    points = kin.Solver(spec).solve(pose)
    bent = flexions(points)

    # Assert
    assert bent["elbow"] > 0
    assert bent["knee"] > 0


@pytest.mark.parametrize("name", ["romanian_deadlift_barbell.json", "single_leg_romanian_deadlift_dumbbell.json",
                                  "bench_press_barbell.json", "chest_press_machine.json"])
def test_bone_lengths_are_constant_across_the_whole_timeline(name):
    # Arrange
    _, solved = frames(load(name))

    # Act
    lengths = {tuple(round(float(np.linalg.norm(p[a] - p[b])), 5)
                     for a, b in (("near_knee", "near_ankle"), ("hip", "near_knee"), ("hip", "shoulder"),
                                  ("shoulder", "near_elbow"), ("near_elbow", "near_wrist"), ("far_knee", "far_ankle")))
               for p in solved}

    # Assert
    assert len(lengths) == 1


def test_standing_feet_stay_planted():
    # Arrange
    _, solved = frames(load("romanian_deadlift_barbell.json"))

    # Act / Assert
    assert all(abs(min(p["heel"][1], p["toe"][1]) - sk.FLOOR_Y) < 1e-6 for p in solved)
    assert len({round(float(p["ankle"][0]), 6) for p in solved}) == 1


@pytest.mark.parametrize("name", ["bench_press_barbell.json", "incline_bench_press_barbell.json", "chest_press_machine.json"])
def test_hips_stay_on_the_support(name):
    # Arrange
    _, solved = frames(load(name))

    # Act
    hips = {tuple(np.round(p["hip"], 6)) for p in solved}

    # Assert
    assert len(hips) == 1


@pytest.mark.parametrize("name", ["bench_press_barbell.json", "chest_press_machine.json"])
def test_fixed_feet_stay_on_the_floor(name):
    # Arrange
    solver, solved = frames(load(name))
    target = solver.contact_target(Contact.NEAR_FOOT)

    # Act / Assert
    assert np.isclose(target[1], sk.FLOOR_Y)
    assert all(np.linalg.norm(p["near_toe"] - target) < 1e-6 for p in solved)


def test_bench_press_bar_path_is_nearly_straight():
    # Arrange
    spec = load("bench_press_barbell.json")
    solver = kin.Solver(spec)

    # Act
    _, _, deviation = scn.fit_line(scn.load_path(spec, solver))

    # Assert: hands given as targets travel in straight lines, no arc behind the head
    assert deviation < 0.35


def test_the_movement_is_centred_in_the_figure_area():
    # Arrange
    _, solved = frames(load("bench_press_barbell.json"))

    # Act
    xs = [float(point[0]) for p in solved for point in p.values()]

    # Assert
    assert abs((min(xs) + max(xs)) / 2 - sk.FIGURE_CENTRE_X) < 1e-6


def test_hanging_body_stays_inside_the_frame():
    # Arrange
    data = json.loads((SAMPLES / "romanian_deadlift_barbell.json").read_text(encoding="utf-8"))
    data["setup"] = {"posture": "hanging", "support": "pull_up_bar"}
    for keyframe in data["keyframes"]:
        keyframe.update({"torso": 90, "near_arm": {"proximal": 90, "distal": 90},
                         "near_leg": {"proximal": 300, "distal": 215}})
    spec = AnimationSpec.model_validate(data)

    # Act
    solver, solved = frames(spec)

    # Assert: the hands stay on the bar, the body fits and no floor is drawn under a pull-up bar
    assert all(np.allclose(p["near_wrist"], solved[0]["near_wrist"]) for p in solved)
    assert min(point[1] for p in solved for point in p.values()) >= kin.FRAME_BOTTOM
    assert not solver.floor_visible


def test_overhead_positions_move_the_scene_down():
    # Arrange
    spec = load("overhead_press_barbell.json")

    # Act
    solver, solved = frames(spec)
    top = max(float(solver.load(p)[1]) for p in solved) + kin.LOAD_MARGIN

    # Assert
    assert solver.floor_y < sk.FLOOR_Y
    assert top <= kin.FRAME_TOP + 1e-6
    assert all(abs(min(p["heel"][1], p["toe"][1]) - solver.floor_y) < 1e-6 for p in solved)
