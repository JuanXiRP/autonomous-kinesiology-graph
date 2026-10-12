# src/rendering/checks.py
# Anatomy, contact, collision and load-path checks over a spec. Every problem is a sentence fed back to
# Gemini, so it names the field to change. Pure, no Manim import.
import re

import numpy as np

from src.domain.animation import AnimationSpec, LoadAttachment, Posture, RigKind, SupportKind, View
from src.rendering import anatomy
from src.rendering import kinematics as kin
from src.rendering import scenery as scn
from src.rendering import skeleton as sk
from src.rendering.skeleton import Pose

# Flexion ranges in degrees, measured relative to the neighbouring segment (independent of the posture)
JOINT_RANGES = {"knee": (-8.0, 155.0), "elbow": (-8.0, 155.0), "hip": (-35.0, 145.0), "shoulder": (-75.0, 190.0)}
BACK_SUPPORT_TOLERANCE = 15.0
MIDFOOT_TOLERANCE = 0.45  # horizontal units the load may drift from mid-foot (~14 cm)
CONTACT_TOLERANCE = 0.08
FLOOR_TOLERANCE = 0.05
PENETRATION_TOLERANCE = 0.06
BAR_RADIUS = 0.08
FRAME_BOTTOM = -3.9
STRAIGHT_PATH_TOLERANCE = 0.15
ARC_PATH_TOLERANCE = 0.25
LOAD_FLOOR_RADIUS = {"barbell": 0.55, "dumbbell": 0.24, "kettlebell": 0.55}

READABLE = {"head": "head", "hip": "hips", "shoulder": "shoulders"}
for _side in ("near", "far"):
    for _joint, _name in (("elbow", "elbow"), ("wrist", "hand"), ("knee", "knee"), ("ankle", "ankle"),
                          ("heel", "heel"), ("toe", "toes")):
        READABLE[f"{_side}_{_joint}"] = f"{_side} {_name}"

def flexions(points: dict[str, np.ndarray], side: str = "near") -> dict[str, float]:
    """Joint flexion in degrees: 0 = straight/neutral, positive = flexed the anatomical way."""
    angle = sk.angle_of
    torso = angle(points["shoulder"] - points["hip"])
    upper_arm = angle(points[f"{side}_elbow"] - points["shoulder"])
    forearm = angle(points[f"{side}_wrist"] - points[f"{side}_elbow"])
    thigh = angle(points[f"{side}_knee"] - points["hip"])
    shank = angle(points[f"{side}_ankle"] - points[f"{side}_knee"])
    return {
        "knee": sk.wrap(thigh - shank, -90.0),
        "elbow": sk.wrap(forearm - upper_arm, -90.0),
        "hip": sk.wrap(180.0 - (torso - thigh), -90.0),
        "shoulder": sk.wrap(upper_arm - torso - 180.0, -150.0),  # wide window: extension reads as negative
    }

def check_joints(points, label: str, sides=("near",), arms: bool = True) -> list[str]:
    problems = []
    for side in sides:
        for joint, value in flexions(points, side).items():
            if not arms and joint in ("elbow", "shoulder"):
                continue
            low, high = JOINT_RANGES[joint]
            if value < low:
                fix = {"knee": "the knee bends backwards", "elbow": "the elbow bends backwards",
                       "hip": "the hip is over-extended (the leg goes too far behind the torso)",
                       "shoulder": "the arm goes too far behind the torso"}[joint]
                problems.append(f"{label}: {side} {joint} {value:.0f}° — {fix}")
            elif value > high:
                problems.append(f"{label}: {side} {joint} flexion {value:.0f}° is beyond the human range (max {high:.0f}°)")
    return problems

# Front view: bends are unsigned (the projection hides their direction), abduction is signed (outward > 0)
FRONT_RANGES = {"knee": (0.0, 155.0), "elbow": (0.0, 155.0), "hip": (-30.0, 70.0), "shoulder": (-40.0, 185.0)}

def front_flexions(points: dict[str, np.ndarray], side: str = "near", seated: bool = False) -> dict[str, float]:
    """Front view: elbow/knee bend, and shoulder/hip abduction measured from the torso's downward axis.
    Seated, the thigh is drawn foreshortened (see FrontSolver._seated_leg): the projection is undone to
    measure the real abduction, and the knee bend is not measurable."""
    up = points["shoulder"] - points["hip"]
    up = up / np.linalg.norm(up)
    outward = np.array([up[1], -up[0]]) * (1.0 if side == "near" else -1.0)

    def abduction(root: str, joint: str, drop: float = 1.0) -> float:
        bone = points[f"{side}_{joint}"] - points[f"{side}_{root}"]
        return float(np.degrees(np.arctan2(np.dot(bone, outward), np.dot(bone, -up) / drop)))

    knee = 0.0 if seated else 180.0 - sk.joint_angle(points[f"{side}_hip"], points[f"{side}_knee"], points[f"{side}_ankle"])
    return {
        "knee": knee,
        "elbow": 180.0 - sk.joint_angle(points[f"{side}_shoulder"], points[f"{side}_elbow"], points[f"{side}_wrist"]),
        "hip": abduction("hip", "knee", kin.SEATED_THIGH_DROP if seated else 1.0),
        "shoulder": sk.wrap(abduction("shoulder", "elbow"), -90.0),
    }

def check_front_joints(points, label: str, sides=("near", "far"), seated: bool = False) -> list[str]:
    problems = []
    for side in sides:
        for joint, value in front_flexions(points, side, seated).items():
            low, high = FRONT_RANGES[joint]
            if value > high:
                what = "abduction" if joint in ("hip", "shoulder") else "bend"
                problems.append(f"{label}: {side} {joint} {what} {value:.0f}° is beyond the human range (max {high:.0f}°)")
            elif value < low:
                problems.append(f"{label}: the {side} {'leg' if joint == 'hip' else 'arm'} crosses the body midline too far "
                                f"({-value:.0f}°); keep it on its own side")
    return problems

def check_back_support(spec: AnimationSpec, torso: float, label: str) -> list[str]:
    setup = spec.setup
    expected = None
    if setup.posture == Posture.SUPINE and setup.support in (SupportKind.FLAT_BENCH, SupportKind.INCLINE_BENCH, SupportKind.DECLINE_BENCH):
        expected = 180.0 - setup.support_angle_deg
    elif setup.posture == Posture.SEATED and setup.support == SupportKind.SEAT_WITH_BACKREST:
        expected = 180.0 - setup.support_angle_deg
    elif setup.posture == Posture.PRONE and setup.support in (SupportKind.FLAT_BENCH, SupportKind.INCLINE_BENCH):
        expected = setup.support_angle_deg
    if expected is None or abs(sk.wrap(torso - expected)) <= BACK_SUPPORT_TOLERANCE:
        return []
    return [f"{label}: torso {torso:.0f} leaves the {setup.support.value.replace('_', ' ')}; "
            f"the body lies on it, use torso ≈ {expected:.0f} (support_angle_deg {setup.support_angle_deg:.0f})"]

def check_contacts(points, solver: kin.Solver, label: str) -> list[str]:
    problems = []
    for contact, (kind, _) in solver.contacts.items():
        side = contact.value.split("_")[0]
        effector = points[f"{side}_{kind}"]
        miss = float(np.linalg.norm(effector - solver.contact_target(contact)))
        if miss > CONTACT_TOLERANCE:
            limb = "hand" if kind == "wrist" else "foot"
            problems.append(f"{label}: the {side} {limb} cannot stay planted ({miss:.2f} units out of reach); "
                            f"keep the {'shoulders' if limb == 'hand' else 'hips'} closer to where it is in the first keyframe")
    return problems

def check_floor(points, solver: kin.Solver, label: str, load: np.ndarray | None, equipment: str) -> list[str]:
    names = [name for name in points if name not in kin.ALIASES]
    if not solver.floor_visible:
        lowest = min(names, key=lambda n: points[n][1])
        if points[lowest][1] < FRAME_BOTTOM:
            return [f"{label}: the {READABLE.get(lowest, lowest)} leaves the bottom of the frame; bend the knees"]
        return []
    problems = []
    lowest = min(names, key=lambda n: points[n][1])
    depth = solver.floor_y - float(points[lowest][1])
    if depth > FLOOR_TOLERANCE:
        problems.append(f"{label}: the {READABLE.get(lowest, lowest)} goes {depth:.2f} units through the floor")
    if load is not None and solver.spec.load_attachment != LoadAttachment.FEET:  # a loaded foot is checked as a foot
        radius = LOAD_FLOOR_RADIUS.get(equipment, 0.12)
        depth = solver.floor_y - (float(load[1]) - radius)
        if depth > FLOOR_TOLERANCE:
            problems.append(f"{label}: the load goes {depth:.2f} units through the floor; raise it")
    return problems

def check_penetration(points, load: np.ndarray | None, attachment: LoadAttachment, pads: list[scn.Part], label: str,
                      solid_bar: bool) -> list[str]:
    """A bar crosses the body's midline, so in side view it can never overlap the torso, the head or a bench.
    Dumbbells, handles and pads move beside the body and legitimately overlap it in projection."""
    if load is None or not solid_bar or attachment not in (LoadAttachment.HANDS, LoadAttachment.FEET):
        return []
    problems = []
    depth = anatomy.depth_inside(load, points["hip"], points["shoulder"], "torso") + BAR_RADIUS
    if depth > PENETRATION_TOLERANCE:
        limb = "near_arm" if attachment == LoadAttachment.HANDS else "near_leg"
        problems.append(f"{label}: the load sinks {depth:.2f} units into the torso; it must stop on the body surface "
                        f"(move the {limb} target out: larger reach or another angle)")
    head_depth = sk.HEAD_RADIUS + BAR_RADIUS - float(np.linalg.norm(load - points["head"]))
    if head_depth > PENETRATION_TOLERANCE:
        problems.append(f"{label}: the load passes through the head; keep it clear of the face")
    for pad in pads:
        a, b = np.array(pad.a), np.array(pad.b)
        segment = b - a
        t = float(np.clip(np.dot(load - a, segment) / max(float(np.dot(segment, segment)), 1e-9), 0.0, 1.0))
        gap = float(np.linalg.norm(load - (a + segment * t))) - pad.width / 2 - BAR_RADIUS
        if gap < -PENETRATION_TOLERANCE:
            problems.append(f"{label}: the load goes through the bench/seat ({-gap:.2f} units)")
            break
    return problems

def check_balance(points, load: np.ndarray | None, label: str) -> list[str]:
    """The load must stay over mid-foot (horizontal distance within MIDFOOT_TOLERANCE)."""
    if load is None:
        return []
    midfoot = float((points["near_heel"][0] + points["near_toe"][0]) / 2)
    offset = float(load[0]) - midfoot
    if abs(offset) <= MIDFOOT_TOLERANCE:
        return []
    where = "in front of" if offset > 0 else "behind"
    fix = ("angle the near_arm back (proximal/distal closer to 250) or push the hips further back"
           if offset > 0 else "bring the shoulders forward over the bar")
    return [f"{label}: the load is {abs(offset):.2f} units {where} mid-foot (max {MIDFOOT_TOLERANCE:.2f}); {fix}"]

def check_rig_path(spec: AnimationSpec, path: list[np.ndarray]) -> list[str]:
    rig = spec.setup.rig
    # A lever turning in depth projects onto the front view as an ellipse, not an arc: check side view only
    if len(path) < 3 or rig in (RigKind.FREE, RigKind.CABLE) or spec.view == View.FRONT:
        return []
    if rig in (RigKind.SMITH, RigKind.MACHINE_LINEAR):
        _, unit, deviation = scn.fit_line(path)
        problems = []
        if deviation > STRAIGHT_PATH_TOLERANCE:
            problems.append(f"the {rig.value.replace('_', ' ')} guides the load in a straight line but the path bends "
                            f"{deviation:.2f} units; give the loaded limb targets on one line")
        if rig == RigKind.SMITH and abs(unit[0]) > 0.17:
            problems.append("a smith machine bar moves vertically; keep the bar straight above its start")
        return problems
    spread = max(float(np.linalg.norm(p - path[0])) for p in path)
    circle = scn.fit_circle(path)
    if spread > 0.3 and circle is not None and circle[2] > ARC_PATH_TOLERANCE and circle[1] < scn.LEVER_RADIUS_RANGE[1]:
        return [f"the machine lever moves the handle on an arc but the path wobbles {circle[2]:.2f} units; "
                f"keep the loaded hand/foot on one smooth arc"]
    return []

def _segment_label(spec: AnimationSpec, t: float) -> str:
    timeline = sk.build_timeline(spec)
    for index in range(len(spec.keyframes)):
        if timeline.arrive[index] - 1e-6 <= t <= timeline.leave[index] + 1e-6:
            return f"keyframe {index}"
    for index in range(len(spec.keyframes) - 1):
        if timeline.leave[index] < t < timeline.arrive[index + 1]:
            return f"between keyframe {index} and {index + 1}"
    return f"keyframe {len(spec.keyframes) - 1}"

def check_spec_geometry(spec: AnimationSpec, equipment: str = "") -> list[str]:
    """Runs every check over the keyframes and over the motion between them."""
    solver = kin.Solver(spec)
    scenery = scn.build_scenery(spec, solver, equipment)
    sides = ("near", "far") if spec.has_independent("far_arm") or spec.has_independent("far_leg") else ("near",)
    problems: list[str] = []
    solid_bar = equipment == "barbell" or spec.setup.rig == RigKind.SMITH

    front = spec.view == View.FRONT
    for index, keyframe in enumerate(spec.keyframes):
        label = f"keyframe {index}"
        points = solver.solve(Pose.from_keyframe(keyframe))
        load = solver.load(points)
        problems += check_contacts(points, solver, label)
        if front:
            problems += check_front_joints(points, label, seated=spec.setup.posture == Posture.SEATED)
            continue
        problems += check_joints(points, label, sides, arms=not solver.arms_follow_load)
        problems += check_back_support(spec, keyframe.torso, label)
        if spec.balance_over_midfoot and spec.setup.posture == Posture.STANDING:
            problems += check_balance(points, load, label)

    # Collisions can also happen between keyframes: report each kind once per transition
    seen = set()
    for t in sk.sample_times(spec, 48):
        label = _segment_label(spec, t)
        points = solver.at(t)
        load = solver.load(points)
        found = check_floor(points, solver, label, load, equipment)
        if not front:
            found += check_penetration(points, load, spec.load_attachment, scenery.pads, label, solid_bar)
        for problem in found:
            # The same issue with another depth inside the same transition is reported once
            key = (label, re.sub(r"[\d.]+", "", problem[len(label):]))
            if key not in seen:
                seen.add(key)
                problems.append(problem)

    problems += check_rig_path(spec, scn.load_path(spec, solver))
    return list(dict.fromkeys(problems))
