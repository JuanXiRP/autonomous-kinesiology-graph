# src/rendering/skeleton.py
# Pure pose and timeline model for the exercise renderer: no Manim import, so it can be unit-tested offline.
# Solving a pose into points lives in kinematics.py, the anatomy checks in checks.py.
import math
from dataclasses import dataclass

import numpy as np

from src.domain.animation import AnimationSpec, Keyframe, LimbPose, MuscleRegion

# Bone lengths in scene units (a standing body is about 5.5 units tall, so 1 unit is about 32 cm)
SCALE = 1.25
BONES = {name: length * SCALE for name, length in
         {"shank": 1.05, "thigh": 1.2, "torso": 1.4, "upper_arm": 0.75, "forearm": 0.7}.items()}
LIMB_BONES = {"arm": ("upper_arm", "forearm"), "leg": ("thigh", "shank")}
HEAD_RADIUS = 0.22 * SCALE
HEAD_DISTANCE = 0.45 * SCALE  # shoulder -> head centre along the torso direction
HEEL_OFFSET = np.array([-0.25, -0.2]) * SCALE  # relative to the ankle, foot pointing right
TOE_OFFSET = np.array([0.4, -0.2]) * SCALE
ANKLE_HEIGHT = -HEEL_OFFSET[1]  # ankle above the floor with the foot flat

FLOOR_Y = -2.8
FIGURE_CENTRE_X = 3.7  # the figure area spans x = 0.3 .. 7.0 (the instruction panel is on the left)

def limb_kind(name: str) -> str:
    return "arm" if name.endswith("arm") else "leg"

def limb_length(name: str) -> float:
    upper, lower = LIMB_BONES[limb_kind(name)]
    return BONES[upper] + BONES[lower]

def direction(angle_deg: float) -> np.ndarray:
    radians = math.radians(angle_deg)
    return np.array([math.cos(radians), math.sin(radians)])

def angle_of(vector: np.ndarray) -> float:
    return math.degrees(math.atan2(vector[1], vector[0]))

def rotate(vector: np.ndarray, angle_deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]])

def wrap(angle_deg: float, low: float = -180.0) -> float:
    """Angle folded into [low, low + 360)."""
    return (angle_deg - low) % 360.0 + low

def joint_angle(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """Interior angle in degrees at `vertex` between the rays towards a and b."""
    u, v = a - vertex, b - vertex
    cosine = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

@dataclass
class LimbState:
    """One limb in a frame: bone angles, or the hand/ankle position relative to the limb root."""
    proximal: float | None = None
    distal: float | None = None
    vector: np.ndarray | None = None
    bend: str | None = None  # front-view IK hint (see animation.Bend)

    @classmethod
    def from_limb_pose(cls, limb: LimbPose, name: str) -> "LimbState":
        bend = limb.bend.value if limb.bend is not None else None
        if limb.target is not None:
            return cls(vector=limb.target.reach * limb_length(name) * direction(limb.target.angle_deg), bend=bend)
        return cls(proximal=limb.proximal, distal=limb.distal, bend=bend)

    @property
    def uses_vector(self) -> bool:
        return self.vector is not None

    def end_vector(self, name: str) -> np.ndarray:
        if self.vector is not None:
            return self.vector
        upper, lower = LIMB_BONES[limb_kind(name)]
        return BONES[upper] * direction(self.proximal) + BONES[lower] * direction(self.distal)

    def lerp(self, other: "LimbState", alpha: float, name: str) -> "LimbState":
        bend = other.bend if alpha >= 0.5 else self.bend
        if not self.uses_vector and not other.uses_vector:
            return LimbState(self.proximal + (other.proximal - self.proximal) * alpha,
                             self.distal + (other.distal - self.distal) * alpha, bend=bend)
        # Hands and feet given as targets travel in a straight line, not along the arc that
        # interpolating joint angles would draw
        a, b = self.end_vector(name), other.end_vector(name)
        return LimbState(vector=a + (b - a) * alpha, bend=bend)

@dataclass
class Pose:
    torso: float
    near_arm: LimbState
    near_leg: LimbState
    far_arm: LimbState | None = None  # None = mirrors the near limb
    far_leg: LimbState | None = None

    def limb(self, name: str) -> LimbState:
        """The limb's state; a mirrored far limb returns the near one."""
        state = getattr(self, name)
        return state if state is not None else getattr(self, name.replace("far_", "near_"))

    @classmethod
    def from_keyframe(cls, keyframe: Keyframe) -> "Pose":
        def state(name):
            limb = keyframe.limb(name)
            return None if limb is None else LimbState.from_limb_pose(limb, name)
        return cls(keyframe.torso, state("near_arm"), state("near_leg"), state("far_arm"), state("far_leg"))

    def lerp(self, other: "Pose", alpha: float) -> "Pose":
        def mix(name):
            mine, theirs = getattr(self, name), getattr(other, name)
            if mine is None and theirs is None:
                return None
            return self.limb(name).lerp(other.limb(name), alpha, name)
        return Pose(self.torso + (other.torso - self.torso) * alpha,
                    mix("near_arm"), mix("near_leg"), mix("far_arm"), mix("far_leg"))

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

def sample_times(spec: AnimationSpec, per_rep: int = 48) -> list[float]:
    """Evenly spaced times over one repetition, keyframe arrivals included."""
    timeline = build_timeline(spec)
    rep = timeline.rep_duration
    times = {rep * i / per_rep for i in range(per_rep)} | set(timeline.arrive)
    return sorted(t for t in times if t < rep)

def peak_activation_time(spec: AnimationSpec) -> float:
    """Time (inside the first repetition) of the keyframe with the highest total muscle tension."""
    timeline = build_timeline(spec)
    totals = [sum(item.level for item in keyframe.muscle_activation) for keyframe in spec.keyframes]
    best = max(range(len(totals)), key=lambda i: totals[i])
    return timeline.arrive[best] + spec.keyframes[best].hold_s / 2
