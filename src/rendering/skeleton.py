# src/rendering/skeleton.py
# Pure geometry for the exercise renderer: no Manim import, so it can be unit-tested offline.
import math
from dataclasses import dataclass

import numpy as np

from src.domain.animation import AnchorKind, AnimationSpec, Keyframe, MuscleRegion

# Bone lengths in scene units (a standing body is about 4.5 units tall)
SCALE = 1.25
BONES = {name: length * SCALE for name, length in
         {"shank": 1.05, "thigh": 1.2, "torso": 1.4, "upper_arm": 0.75, "forearm": 0.7}.items()}
HEAD_RADIUS = 0.22 * SCALE
HEAD_DISTANCE = 0.45 * SCALE  # shoulder -> head centre along the torso direction
HEEL_OFFSET = np.array([-0.25, -0.2]) * SCALE
TOE_OFFSET = np.array([0.4, -0.2]) * SCALE

MIDFOOT_TOLERANCE = 0.45  # horizontal units the load may drift from mid-foot (~14 cm)

FLOOR_Y = -2.8
FEET_ANCHOR_X = 3.3
HANDS_ANCHOR_X = 3.3

ANGLE_NAMES = ("shank", "thigh", "torso", "upper_arm", "forearm")

@dataclass
class Pose:
    shank: float
    thigh: float
    torso: float
    upper_arm: float
    forearm: float
    free_thigh: float | None = None  # non-working leg of single-leg exercises
    free_shank: float | None = None

    @property
    def has_free_leg(self) -> bool:
        return self.free_thigh is not None and self.free_shank is not None

    @classmethod
    def from_keyframe(cls, keyframe: Keyframe) -> "Pose":
        return cls(*(getattr(keyframe, name) for name in ANGLE_NAMES),
                   free_thigh=keyframe.free_thigh, free_shank=keyframe.free_shank)

    def lerp(self, other: "Pose", alpha: float) -> "Pose":
        def mix(a, b):
            return a + (b - a) * alpha
        pose = Pose(*(mix(getattr(self, name), getattr(other, name)) for name in ANGLE_NAMES))
        if self.has_free_leg and other.has_free_leg:
            pose.free_thigh = mix(self.free_thigh, other.free_thigh)
            pose.free_shank = mix(self.free_shank, other.free_shank)
        return pose

def direction(angle_deg: float) -> np.ndarray:
    radians = math.radians(angle_deg)
    return np.array([math.cos(radians), math.sin(radians)])

def rotate(vector: np.ndarray, angle_deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]])

FREE_POINTS = ("free_knee", "free_ankle", "free_heel", "free_toe")

def solve(pose: Pose, anchor: AnchorKind = AnchorKind.FEET, hand_y: float = 2.0) -> dict[str, np.ndarray]:
    """Forward kinematics from the ankle upwards, then placement on the floor or on the bar."""
    ankle = np.zeros(2)
    knee = ankle + BONES["shank"] * direction(pose.shank)
    hip = knee + BONES["thigh"] * direction(pose.thigh)
    shoulder = hip + BONES["torso"] * direction(pose.torso)
    elbow = shoulder + BONES["upper_arm"] * direction(pose.upper_arm)
    wrist = elbow + BONES["forearm"] * direction(pose.forearm)
    points = {
        "ankle": ankle, "knee": knee, "hip": hip, "shoulder": shoulder,
        "elbow": elbow, "wrist": wrist,
        "heel": ankle + HEEL_OFFSET, "toe": ankle + TOE_OFFSET,
        "head": shoulder + HEAD_DISTANCE * direction(pose.torso),
    }
    if pose.has_free_leg:
        free_knee = hip - BONES["thigh"] * direction(pose.free_thigh)
        free_ankle = free_knee - BONES["shank"] * direction(pose.free_shank)
        foot_turn = pose.free_shank - 90.0  # the foot follows the shank
        points.update({
            "free_knee": free_knee, "free_ankle": free_ankle,
            "free_heel": free_ankle + rotate(HEEL_OFFSET, foot_turn),
            "free_toe": free_ankle + rotate(TOE_OFFSET, foot_turn),
        })

    if anchor == AnchorKind.HANDS:
        offset = np.array([HANDS_ANCHOR_X, hand_y]) - wrist
    else:
        # Only the stance leg and the body touch the floor; a lifted free leg never lifts the figure
        lowest = min(point[1] for name, point in points.items() if name not in FREE_POINTS)
        offset = np.array([FEET_ANCHOR_X - ankle[0], FLOOR_Y - lowest])
    return {name: point + offset for name, point in points.items()}

def hand_anchor_y(spec: AnimationSpec) -> float:
    """Height of the overhead bar so a hanging body in the first pose stays just above the floor."""
    relative = solve(Pose.from_keyframe(spec.keyframes[0]), AnchorKind.HANDS, hand_y=0.0)
    lowest = min(point[1] for point in relative.values())
    return FLOOR_Y + 0.6 - lowest

def joint_angle(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """Interior angle in degrees at `vertex` between the rays towards a and b."""
    u, v = a - vertex, b - vertex
    cosine = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

def interior_angles(points: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "knee": joint_angle(points["ankle"], points["knee"], points["hip"]),
        "hip": joint_angle(points["knee"], points["hip"], points["shoulder"]),
        "elbow": joint_angle(points["shoulder"], points["elbow"], points["wrist"]),
    }

def check_pose(points: dict[str, np.ndarray], label: str) -> list[str]:
    """Anatomical sanity checks. Returns human-readable problems (empty list when valid)."""
    problems = []
    angles = interior_angles(points)
    for joint, interior in angles.items():
        flexion = 180.0 - interior
        if flexion > 160.0:
            problems.append(f"{label}: {joint} flexion {flexion:.0f}° is beyond the human range")

    # Facing right: a flexed knee must point forward (to the right of the ankle->hip line)
    ankle, knee, hip = points["ankle"], points["knee"], points["hip"]
    knee_flexion = 180.0 - angles["knee"]
    cross = (hip - ankle)[0] * (knee - ankle)[1] - (hip - ankle)[1] * (knee - ankle)[0]
    if knee_flexion > 8.0 and cross > 0:
        problems.append(f"{label}: the knee bends backwards (it must bend towards the right)")

    if "free_knee" in points:
        free_ankle, free_knee = points["free_ankle"], points["free_knee"]
        free_flexion = 180.0 - joint_angle(free_ankle, free_knee, hip)
        if free_flexion > 160.0:
            problems.append(f"{label}: free knee flexion {free_flexion:.0f}° is beyond the human range")
        cross = (hip - free_ankle)[0] * (free_knee - free_ankle)[1] - (hip - free_ankle)[1] * (free_knee - free_ankle)[0]
        if free_flexion > 8.0 and cross > 0:
            problems.append(f"{label}: the free knee bends the wrong way (free_shank must fold behind free_thigh)")
    return problems

def check_free_leg_clearance(points: dict[str, np.ndarray], label: str) -> list[str]:
    """The lifted leg must stay above the floor."""
    if "free_knee" not in points:
        return []
    lowest = min(points[name][1] for name in FREE_POINTS)
    if lowest < FLOOR_Y - 0.05:
        return [f"{label}: the free leg goes {FLOOR_Y - lowest:.2f} units through the floor; "
                f"lift it (free_thigh/free_shank closer to 0 extends it backwards, 90 hangs it down)"]
    return []

@dataclass
class Timeline:
    arrive: list[float]  # time at which each keyframe is reached
    leave: list[float]   # time at which the pose starts changing again
    rep_duration: float

def build_timeline(spec: AnimationSpec) -> Timeline:
    arrive, leave, t = [], [], 0.0
    for index, keyframe in enumerate(spec.keyframes):
        if index > 0:
            t += keyframe.duration_s
        arrive.append(t)
        t += keyframe.hold_s
        leave.append(t)
    return Timeline(arrive, leave, t)

def total_duration(spec: AnimationSpec) -> float:
    return build_timeline(spec).rep_duration * spec.reps

def smoothstep(x: float) -> float:
    return x * x * (3 - 2 * x)

@dataclass
class Frame:
    pose: Pose
    activation: dict[MuscleRegion, float]
    active_step: int

def _activation(keyframe: Keyframe) -> dict[MuscleRegion, float]:
    return {item.region: item.level for item in keyframe.muscle_activation}

def frame_at(spec: AnimationSpec, t: float) -> Frame:
    """Pose, muscle tension and highlighted step at absolute time t (repetitions loop)."""
    timeline = build_timeline(spec)
    keyframes = spec.keyframes
    if t >= timeline.rep_duration * spec.reps:
        last = keyframes[-1]
        return Frame(Pose.from_keyframe(last), _activation(last), last.active_step)
    t = t % timeline.rep_duration

    for index, keyframe in enumerate(keyframes):
        if timeline.arrive[index] <= t <= timeline.leave[index]:
            return Frame(Pose.from_keyframe(keyframe), _activation(keyframe), keyframe.active_step)

    for index in range(len(keyframes) - 1):
        if timeline.leave[index] < t < timeline.arrive[index + 1]:
            span = timeline.arrive[index + 1] - timeline.leave[index]
            alpha = smoothstep((t - timeline.leave[index]) / span)
            source, target = keyframes[index], keyframes[index + 1]
            before, after = _activation(source), _activation(target)
            blended = {
                region: before.get(region, 0.0) + (after.get(region, 0.0) - before.get(region, 0.0)) * alpha
                for region in set(before) | set(after)
            }
            pose = Pose.from_keyframe(source).lerp(Pose.from_keyframe(target), alpha)
            return Frame(pose, blended, target.active_step)

    last = keyframes[-1]
    return Frame(Pose.from_keyframe(last), _activation(last), last.active_step)

def load_point(points: dict[str, np.ndarray], attachment: str) -> np.ndarray | None:
    if attachment == "hands":
        return points["wrist"]
    if attachment == "shoulders":
        return points["shoulder"]
    return None

def midfoot_x(points: dict[str, np.ndarray]) -> float:
    return float((points["heel"][0] + points["toe"][0]) / 2)

def check_balance(points: dict[str, np.ndarray], attachment: str, label: str) -> list[str]:
    """The load must stay over mid-foot (horizontal distance within MIDFOOT_TOLERANCE)."""
    load = load_point(points, attachment)
    if load is None:
        return []
    offset = float(load[0]) - midfoot_x(points)
    if abs(offset) <= MIDFOOT_TOLERANCE:
        return []
    where = "in front of" if offset > 0 else "behind"
    fix = ("lean the arms back (upper_arm/forearm closer to 250) or push the hips further back"
           if offset > 0 else "bring the shoulders forward over the bar")
    return [f"{label}: the load is {abs(offset):.2f} units {where} mid-foot "
            f"(max {MIDFOOT_TOLERANCE:.2f}); {fix}"]

def check_spec_geometry(spec: AnimationSpec) -> list[str]:
    """Runs the pose checks over every keyframe."""
    problems: list[str] = []
    for index, keyframe in enumerate(spec.keyframes):
        points = solve(Pose.from_keyframe(keyframe), spec.anchor, hand_anchor_y(spec))
        problems.extend(check_pose(points, f"keyframe {index}"))
        if spec.anchor == AnchorKind.FEET:
            problems.extend(check_free_leg_clearance(points, f"keyframe {index}"))
        if spec.balance_over_midfoot and spec.anchor == AnchorKind.FEET:
            problems.extend(check_balance(points, spec.load_attachment.value, f"keyframe {index}"))
    return problems

def peak_activation_time(spec: AnimationSpec) -> float:
    """Time (inside the first repetition) of the keyframe with the highest total muscle tension."""
    timeline = build_timeline(spec)
    totals = [sum(item.level for item in keyframe.muscle_activation) for keyframe in spec.keyframes]
    best = max(range(len(totals)), key=lambda i: totals[i])
    return timeline.arrive[best] + spec.keyframes[best].hold_s / 2
