# src/rendering/scenery.py
# Pure geometry of the scenery and machines, derived from the spec setup and the solved poses: benches,
# seats, backrests, bars, boxes, cable columns, levers and rails. scene.py turns the parts into Manim shapes;
# checks.py uses the pads as obstacles. No Manim import.
import math
from dataclasses import dataclass, field

import numpy as np

from src.domain.animation import AnimationSpec, CableHeight, CableSide, LoadAttachment, Posture, RigKind, SupportKind, View
from src.rendering import kinematics as kin
from src.rendering import skeleton as sk
from src.rendering.skeleton import direction

PAD_THICKNESS = 0.24
FRAME_WIDTH = 0.09
TOP_Y = 3.8
X_LIMITS = (0.35, 6.95)

@dataclass(frozen=True)
class Part:
    """kind: pad (upholstered surface), frame (steel tube), bar (round bar seen end-on: a == b),
    block (box from corner a to corner b), rail (machine guide), pulley (a == b)."""
    kind: str
    a: tuple[float, float]
    b: tuple[float, float]
    width: float

def _part(kind: str, a, b, width: float) -> Part:
    return Part(kind, (float(a[0]), float(a[1])), (float(b[0]), float(b[1])), width)

def _post_to_floor(top, floor: float, width: float = FRAME_WIDTH) -> Part:
    return _part("frame", top, (top[0], floor), width)

@dataclass
class Scenery:
    parts: list[Part] = field(default_factory=list)  # static, drawn behind the body
    pivots: dict[str, np.ndarray] = field(default_factory=dict)   # machine_lever: one per loaded side
    rail: tuple[np.ndarray, np.ndarray] | None = None  # machine_linear / smith: unit direction and a point on it
    pulleys: dict[str, np.ndarray] = field(default_factory=dict)  # cable: "near"/"far" column, or "above"

    @property
    def pulley(self) -> np.ndarray | None:
        return next(iter(self.pulleys.values()), None)

    @property
    def pivot(self) -> np.ndarray | None:
        return self.pivots.get("near")

    @property
    def pads(self) -> list[Part]:
        return [part for part in self.parts if part.kind == "pad"]

# ---- supports --------------------------------------------------------------------------------------

def _supine_pads(points, angle: float, support: SupportKind, floor: float) -> list[Part]:
    along = direction(180.0 - angle)  # towards the head
    back = -kin.torso_front(180.0 - angle)
    surface = points["hip"] + back * kin.HIP_ABOVE_SEAT
    centre = surface + back * PAD_THICKNESS / 2
    torso_reach = sk.BONES["torso"] + sk.HEAD_DISTANCE + 0.2
    parts = []
    if support == SupportKind.INCLINE_BENCH:
        seat_y = points["hip"][1] - kin.HIP_ABOVE_SEAT - PAD_THICKNESS / 2
        seat_a, seat_b = (points["hip"][0] - 0.35, seat_y), (points["hip"][0] + 0.85, seat_y)
        rest_a, rest_b = centre + along * 0.25, centre + along * torso_reach
        parts += [_part("pad", seat_a, seat_b, PAD_THICKNESS), _part("pad", rest_a, rest_b, PAD_THICKNESS)]
        seat_post = (points["hip"][0] + 0.3, seat_y - PAD_THICKNESS / 2)
        strut_top = rest_a + (rest_b - rest_a) * 0.55 + back * PAD_THICKNESS / 2
        parts += [_post_to_floor(seat_post, floor), _post_to_floor(strut_top, floor),
                  _part("frame", (strut_top[0], floor + 0.05), (seat_post[0] + 0.5, floor + 0.05), FRAME_WIDTH)]
        return parts
    feet_end = centre - along * 0.9
    head_end = centre + along * torso_reach
    parts.append(_part("pad", feet_end, head_end, PAD_THICKNESS))
    for end, inward in ((feet_end, along), (head_end, -along)):
        top = end + inward * 0.35 + back * PAD_THICKNESS / 2
        parts.append(_post_to_floor(top, floor))
    return parts

def _prone_pads(points, angle: float, support: SupportKind, floor: float) -> list[Part]:
    along = direction(angle)  # towards the head (prone figures face down with the head to the right)
    chest = kin.torso_front(angle)
    centre = points["hip"] + chest * (kin.CHEST_ABOVE_PAD + PAD_THICKNESS / 2)
    if support == SupportKind.INCLINE_BENCH:
        a, b = centre + along * 0.2, centre + along * (sk.BONES["torso"] + 0.3)
    else:
        a, b = centre - along * 1.1, centre + along * (sk.BONES["torso"] + 0.3)
    parts = [_part("pad", a, b, PAD_THICKNESS)]
    for end, inward in ((a, along), (b, -along)):
        parts.append(_post_to_floor(end + inward * 0.35 + chest * PAD_THICKNESS / 2, floor))
    return parts

def _seat(points, floor: float, length_back: float = 0.5, length_front: float = 0.75) -> list[Part]:
    y = points["hip"][1] - kin.HIP_ABOVE_SEAT - PAD_THICKNESS / 2
    a, b = (points["hip"][0] - length_back, y), (points["hip"][0] + length_front, y)
    return [_part("pad", a, b, PAD_THICKNESS), _post_to_floor((points["hip"][0] + 0.1, y - PAD_THICKNESS / 2), floor)]

def _backrest(points, angle: float, floor: float) -> list[Part]:
    along = direction(180.0 - angle)
    back = -kin.torso_front(180.0 - angle)
    centre = points["hip"] + back * (kin.HIP_ABOVE_SEAT + PAD_THICKNESS / 2)
    a, b = centre + along * 0.2, centre + along * (sk.BONES["torso"] + 0.25)
    bottom = a + back * PAD_THICKNESS / 2
    return [_part("pad", a, b, PAD_THICKNESS), _post_to_floor(bottom, floor)]

def _preacher(points, floor: float) -> list[Part]:
    shoulder, elbow = points["shoulder"], points["near_elbow"]
    unit = (elbow - shoulder) / np.linalg.norm(elbow - shoulder)
    behind = -np.array([-unit[1], unit[0]])  # opposite of the arm's front normal
    offset = behind * (0.19 + PAD_THICKNESS / 2)
    a, b = shoulder + unit * 0.25 + offset, elbow + unit * 0.12 + offset
    low = a if a[1] < b[1] else b
    return [_part("pad", a, b, PAD_THICKNESS), _post_to_floor(low + behind * PAD_THICKNESS / 2, floor)]

def _front_supports(spec: AnimationSpec, solver: kin.Solver) -> list[Part]:
    """Seat and backrest seen from the front: a pad under the pelvis, a board behind the shoulders."""
    support = spec.setup.support
    if support not in (SupportKind.SEAT, SupportKind.SEAT_WITH_BACKREST):
        return []
    points, floor = solver.keyframe_points(0), solver.floor_y
    hip = points["hip"]
    y = hip[1] - kin.HIP_ABOVE_SEAT - PAD_THICKNESS / 2
    parts = []
    if support == SupportKind.SEAT_WITH_BACKREST:
        half = kin.SHOULDER_HALF_WIDTH + 0.1
        top = points["shoulder"][1] + 0.25
        parts.append(_part("block", (hip[0] - half, y), (hip[0] + half, top), 0.0))
    parts.append(_part("pad", (hip[0] - 0.75, y), (hip[0] + 0.75, y), PAD_THICKNESS))
    parts.append(_post_to_floor((hip[0], y - PAD_THICKNESS / 2), floor, 0.14))
    return parts

def support_parts(spec: AnimationSpec, solver: kin.Solver, equipment: str = "") -> list[Part]:
    if spec.view == View.FRONT:
        return _front_supports(spec, solver)
    setup = spec.setup
    points = solver.keyframe_points(0)
    support, angle = setup.support, setup.support_angle_deg
    floor = solver.floor_y
    parts: list[Part] = []
    if setup.posture == Posture.SUPINE and support in (SupportKind.FLAT_BENCH, SupportKind.INCLINE_BENCH, SupportKind.DECLINE_BENCH):
        parts += _supine_pads(points, angle, support, floor)
        if equipment == "barbell" and setup.rig == RigKind.FREE:
            parts += _rack(spec, solver)
    elif setup.posture == Posture.PRONE and support in (SupportKind.FLAT_BENCH, SupportKind.INCLINE_BENCH):
        parts += _prone_pads(points, angle, support, floor)
    elif setup.posture == Posture.BRIDGE and support == SupportKind.FLAT_BENCH:
        top = points["shoulder"][1] - kin.SHOULDER_ABOVE_PAD
        x = points["shoulder"][0]
        parts.append(_part("block", (x - 0.75, floor), (x + 0.2, top - PAD_THICKNESS), 0.0))
        parts.append(_part("pad", (x - 0.75, top - PAD_THICKNESS / 2), (x + 0.2, top - PAD_THICKNESS / 2), PAD_THICKNESS))
    elif setup.posture == Posture.SEATED and support == SupportKind.FLAT_BENCH:
        parts += _seat(points, floor, length_back=1.6, length_front=0.9)
    elif support == SupportKind.SEAT:
        parts += _seat(points, floor)
    elif support == SupportKind.SEAT_WITH_BACKREST:
        parts += _seat(points, floor) + _backrest(points, angle, floor)
    elif support == SupportKind.PREACHER_PAD:
        if setup.posture == Posture.SEATED:
            parts += _seat(points, floor)
        parts += _preacher(points, floor)
    elif support == SupportKind.PULL_UP_BAR:
        bar = points["near_wrist"] + np.array([0.0, 0.04])
        parts.append(_part("frame", bar, (bar[0], TOP_Y), FRAME_WIDTH))
        parts.append(_part("bar", bar, bar, 0.1))
    elif support == SupportKind.PARALLEL_BARS:
        bar = points["near_wrist"] - np.array([0.0, 0.12])
        parts.append(_post_to_floor(bar, floor))
        parts.append(_part("bar", bar, bar, 0.1))
    elif support == SupportKind.BOX:
        heel, toe = points["near_heel"], points["near_toe"]
        parts.append(_part("block", (heel[0] - 0.35, floor), (toe[0] + 0.45, floor + kin.BOX_HEIGHT), 0.0))
    return parts

def _rack(spec: AnimationSpec, solver: kin.Solver) -> list[Part]:
    """Bench-press uprights with a J-hook at the height of the highest bar position, behind the bar."""
    path = load_path(spec, solver)
    if not path:
        return []
    top = max(path, key=lambda p: p[1])
    head = solver.keyframe_points(0)["head"]
    x = min(top[0] - 0.75, head[0] - sk.HEAD_RADIUS - 0.25)
    hook_y = top[1] + 0.1
    return [_part("frame", (x, solver.floor_y), (x, hook_y + 0.5), 0.12),
            _part("frame", (x, hook_y), (x + 0.28, hook_y), 0.08),
            _part("frame", (x + 0.28, hook_y), (x + 0.28, hook_y + 0.14), 0.06)]

# ---- rigs ------------------------------------------------------------------------------------------

def load_path(spec: AnimationSpec, solver: kin.Solver, per_rep: int = 48) -> list[np.ndarray]:
    """Load position over one repetition (empty when nothing is held)."""
    path = []
    for t in sk.sample_times(spec, per_rep):
        point = solver.load(solver.at(t))
        if point is None:
            return []
        path.append(point)
    return path

def fit_line(points: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares line: (centroid, unit direction, max perpendicular deviation)."""
    data = np.array(points)
    centre = data.mean(axis=0)
    _, _, vt = np.linalg.svd(data - centre)
    unit = vt[0]
    normal = np.array([-unit[1], unit[0]])
    deviation = float(np.max(np.abs((data - centre) @ normal))) if len(data) else 0.0
    return centre, unit, deviation

def fit_circle(points: list[np.ndarray]) -> tuple[np.ndarray, float, float] | None:
    """Algebraic (Kasa) circle fit: (centre, radius, max radial residual), None when degenerate."""
    data = np.array(points)
    if len(data) < 3:
        return None
    a = np.column_stack([2 * data[:, 0], 2 * data[:, 1], np.ones(len(data))])
    b = (data ** 2).sum(axis=1)
    try:
        (cx, cy, c), *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    radius_sq = c + cx ** 2 + cy ** 2
    if radius_sq <= 0:
        return None
    centre, radius = np.array([cx, cy]), math.sqrt(radius_sq)
    residual = float(np.max(np.abs(np.linalg.norm(data - centre, axis=1) - radius)))
    return centre, radius, residual

LEVER_RADIUS_RANGE = (0.7, 6.0)

def lever_pivot(path: list[np.ndarray]) -> np.ndarray:
    """Pivot of a lever machine: the centre of the arc the handle draws, or a point above a straight path."""
    circle = fit_circle(path)
    if circle is not None:
        centre, radius, residual = circle
        if LEVER_RADIUS_RANGE[0] <= radius <= LEVER_RADIUS_RANGE[1] and residual < 0.2:
            return centre
    centre, unit, _ = fit_line(path)
    normal = np.array([-unit[1], unit[0]])
    if normal[1] < 0:
        normal = -normal
    distance = 2.0
    if normal[1] > 1e-6:  # keep the pivot inside the frame
        distance = min(distance, (TOP_Y - 0.4 - centre[1]) / normal[1])
    return centre + normal * max(distance, 1.0)

def _clamp_x(x: float) -> float:
    return min(max(x, X_LIMITS[0]), X_LIMITS[1])

def cable_pulley(spec: AnimationSpec, solver: kin.Solver, path: list[np.ndarray]) -> np.ndarray:
    """Side view: one column in front of or behind the figure, or a pulley right above the hands."""
    cable = spec.setup.cable
    first = solver.keyframe_points(0)
    xs = [p[0] for p in path] or [first["near_wrist"][0]]
    ys = [p[1] for p in path] or [first["near_wrist"][1]]
    body_xs = [p[0] for p in first.values()]
    if cable.side == CableSide.FRONT:
        x = max(max(xs), max(body_xs)) + 0.9
    elif cable.side == CableSide.BACK:
        x = min(min(xs), min(body_xs)) - 0.9
    else:
        x = float(np.mean(xs))
    y = max(ys) + 1.6 if cable.side == CableSide.ABOVE else _cable_height(spec, solver, path)
    return np.array([_clamp_x(x), min(y, TOP_Y - 0.2)])

def _cable_height(spec: AnimationSpec, solver: kin.Solver, path: list[np.ndarray]) -> float:
    cable = spec.setup.cable
    first = solver.keyframe_points(0)
    if cable.height == CableHeight.LOW:
        return solver.floor_y + 0.35
    if cable.height == CableHeight.MID:
        return float(first["shoulder"][1])
    return min(max(max(p[1] for p in path), float(first["head"][1])) + 1.0, TOP_Y - 0.2)

def front_cable_pulleys(spec: AnimationSpec, solver: kin.Solver, path: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Front view: a column on each side of the figure (crossover station), or one pulley above."""
    first = solver.keyframe_points(0)
    if spec.setup.cable.side == CableSide.ABOVE:
        return {"above": np.array([float(first["hip"][0]), min(max(p[1] for p in path) + 1.6, TOP_Y - 0.2)])}
    xs = [float(p[0]) for p in first.values()] + [float(p[0]) for p in path]
    y = _cable_height(spec, solver, path)
    return {"near": np.array([_clamp_x(max(xs) + 0.9), y]), "far": np.array([_clamp_x(min(xs) - 0.9), y])}

KNEE_PAD_PIVOT_DROP = 1.2

def front_lever_pivots(spec: AnimationSpec, solver: kin.Solver, path: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Front view: the near lever turns about the centre of the arc its pad draws (the shoulder joint in a
    lateral raise machine); knee pads swing on arms pivoting below the knees. The far pivot is the mirror image."""
    first = solver.keyframe_points(0)
    if spec.load_attachment == LoadAttachment.KNEES:
        near = np.array([float(path[0][0]), float(path[0][1]) - KNEE_PAD_PIVOT_DROP])
    else:
        near = lever_pivot(path)
    pivots = {"near": near}
    if len(solver.loads(first)) > 1:
        axis = (first["shoulder"] - first["hip"]) / np.linalg.norm(first["shoulder"] - first["hip"])
        pivots["far"] = kin.reflect(near, first["hip"], axis)
    return pivots

def build_scenery(spec: AnimationSpec, solver: kin.Solver, equipment: str = "") -> Scenery:
    scenery = Scenery(parts=support_parts(spec, solver, equipment))
    floor = solver.floor_y
    rig = spec.setup.rig
    path = load_path(spec, solver)
    if rig == RigKind.FREE or not path:
        return scenery
    if rig == RigKind.CABLE:
        if spec.view == View.FRONT:
            scenery.pulleys = front_cable_pulleys(spec, solver, path)
        else:
            scenery.pulleys = {"near": cable_pulley(spec, solver, path)}
        for pulley in scenery.pulleys.values():
            if spec.setup.cable.side == CableSide.ABOVE:
                scenery.parts.append(_part("frame", pulley, (pulley[0], TOP_Y), FRAME_WIDTH))
            else:
                scenery.parts.append(_part("rail", (pulley[0], floor), (pulley[0], max(pulley[1] + 0.5, floor + 2.0)), 0.16))
            scenery.parts.append(_part("pulley", pulley, pulley, 0.13))
    elif rig == RigKind.MACHINE_LEVER and spec.view == View.FRONT:
        scenery.pivots = front_lever_pivots(spec, solver, path)
        for pivot in scenery.pivots.values():
            scenery.parts.append(_part("frame", pivot, (pivot[0], floor), 0.14))
            scenery.parts.append(_part("pulley", pivot, pivot, 0.13))
    elif rig == RigKind.MACHINE_LEVER:
        pivot = lever_pivot(path)
        scenery.pivots = {"near": pivot}
        first = solver.keyframe_points(0)
        if pivot[1] > first["shoulder"][1]:
            # Overhead pivot: the lever hangs from a beam on a column behind the seat, weight stack behind it
            column_x = _clamp_x(min(float(p[0]) for p in first.values()) - 0.6)
            scenery.parts.append(_part("frame", pivot, (column_x, pivot[1]), 0.14))
            scenery.parts.append(_part("frame", (column_x, pivot[1]), (column_x, floor), 0.14))
            stack_x = _clamp_x(column_x - 0.35)
        else:
            scenery.parts.append(_part("frame", pivot, (pivot[0], floor), 0.14))
            stack_x = _clamp_x(pivot[0] + (0.5 if pivot[0] >= sk.FIGURE_CENTRE_X else -0.5))
        scenery.parts.append(_part("block", (stack_x - 0.22, floor), (stack_x + 0.22, floor + 1.6), 0.0))
        scenery.parts.append(_part("pulley", pivot, pivot, 0.13))
    else:  # machine_linear, smith
        centre, unit, _ = fit_line(path)
        if rig == RigKind.SMITH:
            unit = np.array([0.0, 1.0])
            x = float(np.mean([p[0] for p in path]))
            scenery.parts.append(_part("rail", (x, floor), (x, TOP_Y - 0.2), 0.12))
        else:
            ends = sorted(((p - centre) @ unit for p in path))
            a, b = centre + unit * (ends[0] - 0.7), centre + unit * (ends[-1] + 0.7)
            scenery.parts.append(_part("rail", a, b, 0.12))
            low = a if a[1] < b[1] else b
            scenery.parts.append(_post_to_floor(low, floor))
        scenery.rail = (unit, centre)
    return scenery

def _pulley_for(spec: AnimationSpec, scenery: Scenery, side: str) -> np.ndarray | None:
    """Side view has one pulley. Front view: cable side 'front' pulls from the same side's column,
    'back' from the opposite column (across the body), 'above' from the overhead pulley."""
    pulleys = scenery.pulleys
    if spec.view != View.FRONT or "above" in pulleys:
        return scenery.pulley
    if spec.setup.cable.side == CableSide.BACK:
        side = "far" if side == "near" else "near"
    return pulleys.get(side)

def moving_parts(spec: AnimationSpec, scenery: Scenery, load: np.ndarray | None,
                 loads: list[tuple[str, np.ndarray]] | None = None) -> list[Part]:
    """Parts that follow the load in each frame: cable, lever arm, carriage or platform."""
    if load is None:
        return []
    rig = spec.setup.rig
    if rig == RigKind.CABLE and scenery.pulleys:
        cables = []
        for side, point in (loads or [("near", load)]):
            pulley = _pulley_for(spec, scenery, side)
            if pulley is not None:
                cables.append(_part("cable", pulley, point, 0.035))
        return cables
    if rig == RigKind.MACHINE_LEVER and scenery.pivots:
        levers = []
        for side, point in (loads or [("near", load)]):
            pivot = scenery.pivots.get(side)
            if pivot is not None:
                levers.append(_part("lever", pivot, point, 0.13))
        return levers
    if rig == RigKind.MACHINE_LINEAR and scenery.rail is not None:
        unit, _ = scenery.rail
        across = np.array([-unit[1], unit[0]])
        return [_part("plate", load - across * 0.55, load + across * 0.55, 0.12)]
    return []
