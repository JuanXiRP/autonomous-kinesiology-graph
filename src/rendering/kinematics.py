# src/rendering/kinematics.py
# Turns a Pose into scene points: forward kinematics for limbs given as angles, two-bone inverse kinematics
# for limbs given as targets, a world root chosen by the posture (planted foot, hips on the seat, hands on
# the bar...), fixed contacts that stay where the first keyframe put them, and automatic framing (centred
# horizontally, moved down when overhead work would leave the frame). FrontSolver handles view = front.
# Pure geometry, no Manim import.
import math

import numpy as np

from src.domain.animation import AnimationSpec, Contact, LoadAttachment, Posture, SupportKind, View
from src.rendering import skeleton as sk
from src.rendering.skeleton import BONES, FLOOR_Y, Pose, direction, rotate

# Heights above the floor, in scene units (1 unit is about 32 cm)
SEAT_HEIGHT = 1.45
BENCH_HEIGHT = 1.4
DECLINE_HIP_RISE = 0.6
BOX_HEIGHT = 1.3
HIP_ABOVE_SEAT = 0.36      # hip joint above the surface the pelvis or lower back rests on
CHEST_ABOVE_PAD = 0.32     # hip joint above the pad in prone positions
SHOULDER_ABOVE_PAD = 0.36  # bridge: shoulder joint above the bench edge or floor
KNEE_ABOVE_FLOOR = 0.15
HAND_ABOVE_FLOOR = 0.12
PULL_UP_CLEARANCE = 0.6
DIP_BAR_HEIGHT = 3.2
TOP_LIMIT_Y = 2.9
FRAME_TOP = 3.75           # highest point a body part or the load may reach on screen
FRAME_BOTTOM = -3.85
LOAD_MARGIN = 0.6          # a barbell plate around the load point
SNAP_DISTANCE = 1.0        # a planted hand/foot this close to the floor in the first keyframe sits on it

# The elbow flexes counter-clockwise (forearm angle = upper arm + flexion), the knee clockwise
FLEX_SIGN = {"arm": -1.0, "leg": 1.0}
HEEL_RAISE_STEPS = (0.0, -15.0, -30.0, -45.0, -60.0, -75.0)
MAX_DORSIFLEXION = 40.0

SIDES = ("near", "far")

def ik(root: np.ndarray, target: np.ndarray, upper: float, lower: float, sign: float) -> tuple[np.ndarray, np.ndarray]:
    """Two-bone IK. Returns (middle joint, end). An unreachable target leaves the limb straight towards it."""
    offset = target - root
    distance = float(np.linalg.norm(offset))
    reach = min(max(distance, abs(upper - lower) + 1e-6), upper + lower - 1e-9)
    base = math.degrees(math.atan2(offset[1], offset[0])) if distance > 1e-9 else 270.0
    cosine = (upper ** 2 + reach ** 2 - lower ** 2) / (2 * upper * reach)
    spread = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    middle = root + upper * direction(base + sign * spread)
    towards = target - middle
    norm = float(np.linalg.norm(towards))
    end = middle + lower * (towards / norm if norm > 1e-9 else direction(base))
    return middle, end

def torso_front(torso_deg: float) -> np.ndarray:
    """Unit vector towards the chest (the figure faces right when standing)."""
    d = direction(torso_deg)
    return np.array([d[1], -d[0]])

def foot_turn(knee: np.ndarray, ankle: np.ndarray) -> float:
    """Neutral foot rotation: the sole at a right angle to the shank."""
    return sk.angle_of(ankle - knee) + 90.0

def set_foot(points: dict, side: str, turn_deg: float) -> None:
    ankle = points[f"{side}_ankle"]
    points[f"{side}_heel"] = ankle + rotate(sk.HEEL_OFFSET, turn_deg)
    points[f"{side}_toe"] = ankle + rotate(sk.TOE_OFFSET, turn_deg)

def limb_points(root: np.ndarray, state: sk.LimbState, name: str) -> tuple[np.ndarray, np.ndarray]:
    kind = sk.limb_kind(name)
    upper, lower = (BONES[bone] for bone in sk.LIMB_BONES[kind])
    if state.uses_vector:
        return ik(root, root + state.vector, upper, lower, FLEX_SIGN[kind])
    middle = root + upper * direction(state.proximal)
    return middle, middle + lower * direction(state.distal)

# Short names kept for force anchors, plus the per-side shoulder/hip joints, which coincide in side view
ALIASES = {"elbow": "near_elbow", "wrist": "near_wrist", "knee": "near_knee", "ankle": "near_ankle",
           "heel": "near_heel", "toe": "near_toe", "near_shoulder": "shoulder", "far_shoulder": "shoulder",
           "near_hip": "hip", "far_hip": "hip"}

def load_point(points: dict[str, np.ndarray], attachment: LoadAttachment | str, torso_deg: float | None = None) -> np.ndarray | None:
    attachment = LoadAttachment(attachment)
    if attachment == LoadAttachment.HANDS:
        return points["near_wrist"]
    if attachment == LoadAttachment.SHOULDERS:
        # Bar resting on the upper back, just behind and above the shoulder joint
        torso = torso_deg if torso_deg is not None else sk.angle_of(points["shoulder"] - points["hip"])
        return points["shoulder"] - torso_front(torso) * 0.2 + direction(torso) * 0.06
    if attachment == LoadAttachment.FEET:
        return (points["near_heel"] + points["near_toe"]) / 2
    if attachment == LoadAttachment.ELBOWS:
        return points["near_elbow"]
    if attachment == LoadAttachment.KNEES:
        return points["near_knee"]
    if attachment == LoadAttachment.HIPS:
        torso = torso_deg if torso_deg is not None else sk.angle_of(points["shoulder"] - points["hip"])
        return points["hip"] + torso_front(torso) * 0.34 * sk.SCALE
    return None

class Solver:
    """Solves poses of one spec into scene points. The world root, the fixed contacts and the framing are
    computed once from the spec so every frame of the animation shares them. Front-view specs get a
    FrontSolver (Solver(spec) picks it)."""

    def __new__(cls, spec: AnimationSpec):
        if cls is Solver and spec.view == View.FRONT:
            return super().__new__(FrontSolver)
        return super().__new__(cls)

    def __init__(self, spec: AnimationSpec):
        self.spec = spec
        self.setup = spec.setup
        self.shift = np.zeros(2)
        self.hand_y = 0.0
        self.contacts: dict[Contact, tuple[str, np.ndarray]] = {}
        self.independent = {name: spec.has_independent(name) for name in ("far_arm", "far_leg")}
        self.arms_follow_load = self._arms_follow_load()

        first = Pose.from_keyframe(spec.keyframes[0])
        if self.setup.posture == Posture.HANGING:
            self.hand_y = self._hanging_height()
        self.contacts = self._fixed_contacts(first)
        self.shift = np.array([self._frame_offset(), self._vertical_offset()])

    # ---- public API ------------------------------------------------------------------------------

    def solve(self, pose: Pose) -> dict[str, np.ndarray]:
        points = self._solve_world(pose)
        points = {name: point + self.shift for name, point in points.items()}
        for alias, name in ALIASES.items():
            points.setdefault(alias, points[name])
        return points

    def keyframe_points(self, index: int) -> dict[str, np.ndarray]:
        return self.solve(Pose.from_keyframe(self.spec.keyframes[index]))

    def at(self, t: float) -> dict[str, np.ndarray]:
        return self.solve(sk.frame_at(self.spec, t).pose)

    def load(self, points: dict[str, np.ndarray]) -> np.ndarray | None:
        loads = self.loads(points)
        return loads[0][1] if loads else None

    def loads(self, points: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
        """(side, position) of every load drawn. In side view the far hand hides behind the near one."""
        point = load_point(points, self.spec.load_attachment)
        return [] if point is None else [("near", point)]

    @property
    def floor_y(self) -> float:
        """Floor height on screen (the whole scene moves down when a tall movement would leave the frame)."""
        return FLOOR_Y + float(self.shift[1])

    @property
    def floor_visible(self) -> bool:
        return self.setup.posture != Posture.HANGING or self.setup.support == SupportKind.PARALLEL_BARS

    def contact_target(self, contact: Contact) -> np.ndarray:
        """World position (scene coordinates) a fixed hand/foot is held at."""
        _, point = self.contacts[contact]
        return point + self.shift

    # ---- pipeline --------------------------------------------------------------------------------

    def _relative(self, pose: Pose) -> dict[str, np.ndarray]:
        """Points with the hip at the origin; no contact constraints yet."""
        hip = np.zeros(2)
        shoulder = hip + BONES["torso"] * direction(pose.torso)
        points = {"hip": hip, "shoulder": shoulder, "head": shoulder + sk.HEAD_DISTANCE * direction(pose.torso)}
        for side in SIDES:
            points[f"{side}_elbow"], points[f"{side}_wrist"] = limb_points(shoulder, pose.limb(f"{side}_arm"), f"{side}_arm")
            points[f"{side}_knee"], points[f"{side}_ankle"] = limb_points(hip, pose.limb(f"{side}_leg"), f"{side}_leg")
            flat = self.setup.posture == Posture.STANDING and (side == "near" or not self.independent["far_leg"])
            set_foot(points, side, 0.0 if flat else foot_turn(points[f"{side}_knee"], points[f"{side}_ankle"]))
        return points

    def _root(self) -> tuple[str, np.ndarray]:
        """The body point that stays fixed in the world for this posture, and where it sits."""
        posture, support = self.setup.posture, self.setup.support
        on_bench = support in (SupportKind.FLAT_BENCH, SupportKind.INCLINE_BENCH, SupportKind.DECLINE_BENCH)
        if posture == Posture.STANDING:
            lift = BOX_HEIGHT if support == SupportKind.BOX else 0.0
            return "near_ankle", np.array([0.0, FLOOR_Y + sk.ANKLE_HEIGHT + lift])
        if posture == Posture.KNEELING:
            return "near_knee", np.array([0.0, FLOOR_Y + KNEE_ABOVE_FLOOR])
        if posture == Posture.PLANK:
            return "near_toe", np.array([0.0, FLOOR_Y])
        if posture == Posture.HANGING:
            return "near_wrist", np.array([0.0, self.hand_y])
        if posture == Posture.BRIDGE:
            base = BENCH_HEIGHT if on_bench else 0.0
            return "shoulder", np.array([0.0, FLOOR_Y + base + SHOULDER_ABOVE_PAD])
        if posture == Posture.SEATED:
            base = 0.0 if support == SupportKind.NONE else SEAT_HEIGHT
            return "hip", np.array([0.0, FLOOR_Y + base + HIP_ABOVE_SEAT])
        if posture == Posture.PRONE:
            base = {SupportKind.FLAT_BENCH: BENCH_HEIGHT, SupportKind.INCLINE_BENCH: SEAT_HEIGHT}.get(support, 0.0)
            return "hip", np.array([0.0, FLOOR_Y + base + CHEST_ABOVE_PAD])
        # supine
        base = {SupportKind.FLAT_BENCH: BENCH_HEIGHT, SupportKind.INCLINE_BENCH: SEAT_HEIGHT,
                SupportKind.DECLINE_BENCH: BENCH_HEIGHT + DECLINE_HIP_RISE}.get(support, 0.0)
        return "hip", np.array([0.0, FLOOR_Y + base + HIP_ABOVE_SEAT])

    def _place(self, pose: Pose) -> dict[str, np.ndarray]:
        relative = self._relative(pose)
        root_name, root_world = self._root()
        offset = root_world - relative[root_name]
        return {name: point + offset for name, point in relative.items()}

    def _solve_world(self, pose: Pose) -> dict[str, np.ndarray]:
        points = self._place(pose)
        root_name, _ = self._root()
        if self.arms_follow_load:
            self._hold_bar(points, pose)
        else:
            self._place_arms(points, root_name)
        self._place_legs(points, root_name)
        return points

    # ---- hooks the front view overrides ----------------------------------------------------------

    def _arms_follow_load(self) -> bool:
        """A bar on the back or across the hips is held by both hands: the arms follow the bar."""
        return self.spec.load_attachment in (LoadAttachment.SHOULDERS, LoadAttachment.HIPS)

    def _limb_root(self, points: dict, side: str, kind: str) -> np.ndarray:
        return points["shoulder"] if kind == "arm" else points["hip"]

    def _solve_ik(self, points: dict, side: str, kind: str, target: np.ndarray, bend: str | None = None):
        upper, lower = (BONES[bone] for bone in sk.LIMB_BONES[kind])
        return ik(self._limb_root(points, side, kind), target, upper, lower, FLEX_SIGN[kind])

    def _mirror_far(self, points: dict, kind: str) -> None:
        """A bilateral far limb coincides with the near one in side view (drawn slightly offset)."""
        joints = ("elbow", "wrist") if kind == "arm" else ("knee", "ankle", "heel", "toe")
        for joint in joints:
            points[f"far_{joint}"] = points[f"near_{joint}"].copy()

    def _set_contact_foot(self, points: dict, side: str) -> None:
        set_foot(points, side, foot_turn(points[f"{side}_knee"], points[f"{side}_ankle"]))

    # ---- limbs -----------------------------------------------------------------------------------

    def _place_arms(self, points: dict, root_name: str) -> None:
        for side in SIDES:
            if side == "far" and not self.independent["far_arm"]:
                self._mirror_far(points, "arm")
                continue
            contact = Contact(f"{side}_hand")
            if contact in self.contacts and root_name != f"{side}_wrist":
                _, target = self.contacts[contact]
                points[f"{side}_elbow"], points[f"{side}_wrist"] = self._solve_ik(points, side, "arm", target)

    def _place_legs(self, points: dict, root_name: str) -> None:
        for side in SIDES:
            if side == "far" and not self.independent["far_leg"]:
                self._mirror_far(points, "leg")
                continue
            contact = Contact(f"{side}_foot")
            if contact in self.contacts and root_name not in (f"{side}_ankle", f"{side}_toe", f"{side}_knee"):
                self._apply_foot_contact(points, side, *self.contacts[contact])
            elif not (self.setup.posture == Posture.STANDING and side == "near") and root_name != f"{side}_toe":
                self._clear_floor(points, side)

    def _hold_bar(self, points: dict, pose: Pose) -> None:
        """Both hands on the bar. On the back the elbows point down (the arm is abducted in 3D, so the
        side-view projection may fold past the 2D anatomical limit); across the hips they bend normally."""
        target = load_point(points, self.spec.load_attachment, pose.torso)
        upper, lower = BONES["upper_arm"], BONES["forearm"]
        if self.spec.load_attachment == LoadAttachment.SHOULDERS:
            options = [ik(points["shoulder"], target, upper, lower, sign) for sign in (-1.0, 1.0)]
            elbow, wrist = min(options, key=lambda option: option[0][1])
        else:
            elbow, wrist = ik(points["shoulder"], target, upper, lower, FLEX_SIGN["arm"])
        for side in SIDES:
            points[f"{side}_elbow"], points[f"{side}_wrist"] = elbow.copy(), wrist.copy()

    # ---- contacts --------------------------------------------------------------------------------

    def _fixed_contacts(self, first: Pose) -> dict[Contact, tuple[str, np.ndarray]]:
        points = self._place(first)
        contacts = {}
        for contact in self.setup.fixed_contacts:
            side, part = contact.value.split("_")
            if part == "hand":
                wrist = points[f"{side}_wrist"].copy()
                if abs(wrist[1] - FLOOR_Y) < SNAP_DISTANCE:
                    wrist[1] = FLOOR_Y + HAND_ABOVE_FLOOR
                contacts[contact] = ("wrist", wrist)
            else:
                ankle = points[f"{side}_ankle"]
                if abs(ankle[1] - sk.ANKLE_HEIGHT - FLOOR_Y) < SNAP_DISTANCE:
                    # Planted on the floor: the ball of the foot is the fixed point, the heel may rise
                    contacts[contact] = ("toe", np.array([ankle[0] + sk.TOE_OFFSET[0], FLOOR_Y]))
                else:
                    contacts[contact] = ("ankle", ankle.copy())
        return contacts

    def _apply_foot_contact(self, points: dict, side: str, kind: str, target: np.ndarray) -> None:
        hip, thigh, shank = self._limb_root(points, side, "leg"), BONES["thigh"], BONES["shank"]
        if kind == "ankle":
            points[f"{side}_knee"], points[f"{side}_ankle"] = self._solve_ik(points, side, "leg", target)
            self._set_contact_foot(points, side)
            return
        best = None
        for turn in HEEL_RAISE_STEPS:
            ankle = target - rotate(sk.TOE_OFFSET, turn)
            knee, reached = ik(hip, ankle, thigh, shank, FLEX_SIGN["leg"])
            miss = float(np.linalg.norm(reached - ankle))
            dorsiflexion = sk.wrap(turn - foot_turn(knee, reached))
            score = (miss > 0.02, dorsiflexion > MAX_DORSIFLEXION, miss)
            if best is None or score < best[0]:
                best = (score, turn, knee, reached)
            if score[:2] == (False, False):
                break
        _, turn, knee, ankle = best
        points[f"{side}_knee"], points[f"{side}_ankle"] = knee, ankle
        set_foot(points, side, turn)

    def _clear_floor(self, points: dict, side: str) -> None:
        """A foot that is not planted points its toes down (plantar flexion) until it clears the floor."""
        if self.setup.posture == Posture.HANGING and not self.floor_visible:
            return
        neutral = foot_turn(points[f"{side}_knee"], points[f"{side}_ankle"])
        for step in range(7):
            set_foot(points, side, neutral - 15.0 * step)
            if min(points[f"{side}_heel"][1], points[f"{side}_toe"][1]) >= FLOOR_Y - 1e-6:
                return
        set_foot(points, side, neutral)

    # ---- placement -------------------------------------------------------------------------------

    def _sampled(self) -> list[dict[str, np.ndarray]]:
        return [self._solve_world(sk.frame_at(self.spec, t).pose) for t in sk.sample_times(self.spec, 24)]

    def _hanging_height(self) -> float:
        """Bar height: the hanging body stays above the floor (parallel bars sit at a fixed height)."""
        lowest = min(min(p[1] for p in self._place(sk.frame_at(self.spec, t).pose).values())
                     for t in sk.sample_times(self.spec, 24))
        needed = FLOOR_Y + PULL_UP_CLEARANCE - lowest
        if self.setup.support == SupportKind.PARALLEL_BARS:
            needed = max(needed, FLOOR_Y + DIP_BAR_HEIGHT)
        return min(needed, TOP_LIMIT_Y)

    def _vertical_offset(self) -> float:
        """Moves everything down (floor included) when an overhead position would leave the frame."""
        frames = self._sampled()
        top = max(float(p[1]) for frame in frames for p in frame.values())
        loads = [load_point(frame, self.spec.load_attachment) for frame in frames]
        top = max([top] + [float(load[1]) + LOAD_MARGIN for load in loads if load is not None])
        bottom = min(float(p[1]) for frame in frames for p in frame.values())
        if top <= FRAME_TOP:
            return 0.0
        return max(FRAME_TOP - top, FRAME_BOTTOM - min(bottom, FLOOR_Y))

    def _frame_offset(self) -> float:
        """Horizontal shift that centres the whole movement in the figure area."""
        xs = [float(p[0]) for frame in self._sampled() for p in frame.values()]
        return sk.FIGURE_CENTRE_X - (min(xs) + max(xs)) / 2


# ---- front view --------------------------------------------------------------------------------------

SHOULDER_HALF_WIDTH = 0.48 * sk.SCALE
HIP_HALF_WIDTH = 0.24 * sk.SCALE
FRONT_HEEL = np.array([-0.07, -0.2]) * sk.SCALE  # inner edge of the screen-right foot seen from the front
FRONT_TOE = np.array([0.13, -0.2]) * sk.SCALE    # outer edge
DEFAULT_BEND = {"arm": "down", "leg": "out"}
# Seated and seen from the front, the thigh points at the camera: its projection is short and spreads
# sideways with hip abduction. This drop keeps the feet just on the floor below a standard seat.
SEATED_THIGH_DROP = 0.17

def reflect(point: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Mirror image of `point` across the line through `origin` along the unit vector `axis`."""
    foot = origin + axis * float(np.dot(point - origin, axis))
    return 2 * foot - point

class FrontSolver(Solver):
    """The figure faces the camera: near limbs on the screen right, a mirrored far limb reflected across the
    body midline, shoulder and hip joints spread to the sides, and elbows/knees bent as the `bend` hint says."""

    def __init__(self, spec: AnimationSpec):
        self.spec, self.setup = spec, spec.setup
        self.independent = {name: spec.has_independent(name) for name in ("far_arm", "far_leg")}
        self.stance = "near"
        self.stance = self._stance_side()
        super().__init__(spec)

    def _arms_follow_load(self) -> bool:
        return False  # front view: the arms keep the poses the spec gives

    def _stance_side(self) -> str:
        """Single-leg work: the leg whose ankle moves least (relative to the pelvis) stands on the floor."""
        if not self.independent["far_leg"]:
            return "near"
        poses = [self._relative(Pose.from_keyframe(keyframe)) for keyframe in self.spec.keyframes]

        def travel(side: str) -> float:
            ankles = [pose[f"{side}_ankle"] - pose["hip"] for pose in poses]
            return max(float(np.linalg.norm(a - ankles[0])) for a in ankles)

        return "far" if travel("far") < travel("near") - 1e-6 else "near"

    @staticmethod
    def _axis(points: dict) -> np.ndarray:
        up = points["shoulder"] - points["hip"]
        return up / np.linalg.norm(up)

    def _relative(self, pose: Pose) -> dict[str, np.ndarray]:
        up = direction(pose.torso)
        right = np.array([up[1], -up[0]])
        hip = np.zeros(2)
        shoulder = hip + BONES["torso"] * up
        points = {"hip": hip, "shoulder": shoulder, "head": shoulder + sk.HEAD_DISTANCE * up,
                  "near_shoulder": shoulder + right * SHOULDER_HALF_WIDTH, "far_shoulder": shoulder - right * SHOULDER_HALF_WIDTH,
                  "near_hip": hip + right * HIP_HALF_WIDTH, "far_hip": hip - right * HIP_HALF_WIDTH}
        for side in SIDES:
            for kind, joints in (("arm", ("elbow", "wrist")), ("leg", ("knee", "ankle"))):
                if side == "far" and not self.independent[f"far_{kind}"]:
                    continue
                state = pose.limb(f"{side}_{kind}")
                root = self._limb_root(points, side, kind)
                if kind == "leg" and self.setup.posture == Posture.SEATED:
                    middle, end = self._seated_leg(points, side, state)
                elif state.uses_vector:
                    middle, end = self._solve_ik(points, side, kind, root + state.vector, state.bend)
                else:
                    upper, lower = (BONES[bone] for bone in sk.LIMB_BONES[kind])
                    middle = root + upper * direction(state.proximal)
                    end = middle + lower * direction(state.distal)
                points[f"{side}_{joints[0]}"], points[f"{side}_{joints[1]}"] = middle, end
        for kind in ("arm", "leg"):
            if not self.independent[f"far_{kind}"]:
                self._mirror_far(points, kind)
        for side in SIDES:
            self._set_contact_foot(points, side)
        return points

    def _seated_leg(self, points: dict, side: str, state: sk.LimbState) -> tuple[np.ndarray, np.ndarray]:
        """Thigh pointing at the camera, foreshortened: the given direction only says how far the knee
        swings out (abduction, measured from straight down); the shank hangs from the knee as given."""
        root = self._limb_root(points, side, "leg")
        axis = self._axis(points)
        outward = np.array([axis[1], -axis[0]]) * (1.0 if side == "near" else -1.0)
        thigh_direction = state.vector if state.uses_vector else direction(state.proximal)
        abduction = math.atan2(float(np.dot(thigh_direction, outward)), float(np.dot(thigh_direction, -axis)))
        knee = root + BONES["thigh"] * (math.sin(abduction) * outward - axis * SEATED_THIGH_DROP * math.cos(abduction))
        shank = direction(state.distal) if not state.uses_vector else -axis
        return knee, knee + BONES["shank"] * shank

    def _root(self) -> tuple[str, np.ndarray]:
        posture = self.setup.posture
        if posture == Posture.STANDING:
            return f"{self.stance}_ankle", np.array([0.0, FLOOR_Y + sk.ANKLE_HEIGHT])
        if posture == Posture.KNEELING:
            return f"{self.stance}_knee", np.array([0.0, FLOOR_Y + KNEE_ABOVE_FLOOR])
        return super()._root()

    def _limb_root(self, points: dict, side: str, kind: str) -> np.ndarray:
        return points[f"{side}_shoulder"] if kind == "arm" else points[f"{side}_hip"]

    def _solve_ik(self, points: dict, side: str, kind: str, target: np.ndarray, bend: str | None = None):
        upper, lower = (BONES[bone] for bone in sk.LIMB_BONES[kind])
        root = self._limb_root(points, side, kind)
        options = [ik(root, target, upper, lower, sign) for sign in (-1.0, 1.0)]
        axis = self._axis(points)
        outward = np.array([axis[1], -axis[0]]) * (1.0 if side == "near" else -1.0)
        bend = bend or DEFAULT_BEND[kind]
        score = {"out": lambda m: float(np.dot(m - root, outward)), "in": lambda m: -float(np.dot(m - root, outward)),
                 "up": lambda m: float(m[1]), "down": lambda m: -float(m[1])}[bend]
        return max(options, key=lambda option: score(option[0]))

    def _mirror_far(self, points: dict, kind: str) -> None:
        axis = self._axis(points)
        joints = ("elbow", "wrist") if kind == "arm" else ("knee", "ankle", "heel", "toe")
        for joint in joints:
            if f"near_{joint}" in points:
                points[f"far_{joint}"] = reflect(points[f"near_{joint}"], points["hip"], axis)

    def _set_contact_foot(self, points: dict, side: str) -> None:
        """Feet seen from the front stay flat; the far foot is the mirror image of the near one."""
        sign = 1.0 if side == "near" else -1.0
        ankle = points[f"{side}_ankle"]
        points[f"{side}_heel"] = ankle + FRONT_HEEL * np.array([sign, 1.0])
        points[f"{side}_toe"] = ankle + FRONT_TOE * np.array([sign, 1.0])

    def _clear_floor(self, points: dict, side: str) -> None:
        return

    def loads(self, points: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
        """Both hands (or feet) carry a load when they mirror each other; single-leg work loads the moving leg."""
        attachment = self.spec.load_attachment
        if attachment in (LoadAttachment.HANDS, LoadAttachment.ELBOWS):
            joint = "wrist" if attachment == LoadAttachment.HANDS else "elbow"
            sides = ["near"] + ([] if self.independent["far_arm"] else ["far"])
            return [(side, points[f"{side}_{joint}"]) for side in sides]
        if attachment in (LoadAttachment.FEET, LoadAttachment.KNEES):
            if self.independent["far_leg"]:
                moving = "far" if self.stance == "near" else "near"
                sides = [moving]
            else:
                sides = ["near", "far"]
            if attachment == LoadAttachment.KNEES:
                return [(side, points[f"{side}_knee"]) for side in sides]
            return [(side, (points[f"{side}_heel"] + points[f"{side}_toe"]) / 2) for side in sides]
        point = load_point(points, attachment)
        return [] if point is None else [("near", point)]

    def _fixed_contacts(self, first: Pose) -> dict[Contact, tuple[str, np.ndarray]]:
        """Planted feet sit flat on the floor in front view (no heel raise to solve). Seated, the feet simply
        hang from the knees (the foreshortened thigh cannot be solved with IK), so they are not glued."""
        contacts = super()._fixed_contacts(first)
        if self.setup.posture == Posture.SEATED:
            contacts = {c: v for c, v in contacts.items() if not c.value.endswith("foot")}
        points = self._place(first)
        for contact, (kind, _) in list(contacts.items()):
            if kind == "toe":
                side = contact.value.split("_")[0]
                ankle = points[f"{side}_ankle"].copy()
                ankle[1] = FLOOR_Y + sk.ANKLE_HEIGHT
                contacts[contact] = ("ankle", ankle)
        return contacts
