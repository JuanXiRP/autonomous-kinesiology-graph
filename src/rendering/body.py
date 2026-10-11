# src/rendering/body.py
# Anatomical vector body for the Manim scene: tapered limb silhouettes with muscle bellies that follow
# the outline and tint from neutral grey to green with tension. Geometry helpers are pure (no Manim).
import math
from dataclasses import dataclass

import numpy as np
from manim import Circle, Polygon, VMobject

from src.rendering import palette as art
from src.domain.animation import MuscleRegion
from src.rendering import skeleton as sk
from src.rendering.layout import Capsule

S = sk.SCALE

SKIN = "#D9E0DC"
SKIN_FAR = "#E9EEEB"
OUTLINE = "#7C8A83"
OUTLINE_FAR = "#AEB9B3"
MUSCLE_IDLE = "#C4CFC9"
FIBRE = "#9EADA6"

@dataclass(frozen=True)
class Profile:
    """Half-widths of a limb along its length (t=0 at the first joint, t=1 at the second).
    Each side is (start, belly, end, belly_position)."""
    front: tuple[float, float, float, float]
    back: tuple[float, float, float, float]

# Front = the side the figure faces (see front_normal); values in unscaled body units
PROFILES = {
    "shank": Profile(front=(0.09, 0.11, 0.13, 0.6), back=(0.08, 0.22, 0.14, 0.62)),
    "thigh": Profile(front=(0.13, 0.25, 0.21, 0.55), back=(0.13, 0.22, 0.24, 0.6)),
    "torso": Profile(front=(0.22, 0.27, 0.30, 0.8), back=(0.27, 0.25, 0.31, 0.78)),
    "upper_arm": Profile(front=(0.11, 0.135, 0.085, 0.5), back=(0.12, 0.15, 0.085, 0.4)),
    "forearm": Profile(front=(0.09, 0.115, 0.06, 0.25), back=(0.085, 0.105, 0.06, 0.3)),
    "neck": Profile(front=(0.09, 0.085, 0.08, 0.5), back=(0.10, 0.09, 0.08, 0.5)),
}

# region -> (segment, t0, t1, side, depth): side +1 front, -1 back, 0 both; depth = share of the half-width
MUSCLES = {
    MuscleRegion.CALVES: ("shank", 0.35, 0.95, -1, 0.85),
    MuscleRegion.QUADRICEPS: ("thigh", 0.08, 0.92, 1, 0.85),
    MuscleRegion.HAMSTRINGS: ("thigh", 0.10, 0.85, -1, 0.85),
    MuscleRegion.ABS: ("torso", 0.08, 0.60, 1, 0.75),
    MuscleRegion.CHEST: ("torso", 0.62, 0.95, 1, 0.85),
    MuscleRegion.LOWER_BACK: ("torso", 0.05, 0.50, -1, 0.70),
    MuscleRegion.UPPER_BACK: ("torso", 0.55, 1.00, -1, 0.85),
    MuscleRegion.LATS: ("torso", 0.35, 0.80, -1, 0.45),
    MuscleRegion.SHOULDERS: ("upper_arm", -0.05, 0.38, 0, 0.95),
    MuscleRegion.BICEPS: ("upper_arm", 0.30, 0.90, 1, 0.85),
    MuscleRegion.TRICEPS: ("upper_arm", 0.30, 0.90, -1, 0.85),
    MuscleRegion.FOREARMS: ("forearm", 0.05, 0.75, 0, 0.85),
}

def blend(c1_hex: str, c2_hex: str, t: float) -> str:
    c1 = np.array([int(c1_hex[i:i + 2], 16) for i in (1, 3, 5)])
    c2 = np.array([int(c2_hex[i:i + 2], 16) for i in (1, 3, 5)])
    c = np.clip((1 - t) * c1 + t * c2, 0, 255).astype(int)
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"

def muscle_color(level: float) -> str:
    if level <= 0.5:
        return blend(MUSCLE_IDLE, art.GLOW_GREEN, level * 2)
    return blend(art.GLOW_GREEN, art.PRIMARY_GREEN, (level - 0.5) * 2)

def front_normal(direction: np.ndarray, segment: str) -> np.ndarray:
    """Unit vector pointing to the front of the body (the figure faces right)."""
    if segment in ("upper_arm", "forearm"):  # arm vectors point downwards when hanging
        return np.array([-direction[1], direction[0]])
    return np.array([direction[1], -direction[0]])

def half_width(side: tuple[float, float, float, float], t: float) -> float:
    """Smooth width profile through start -> belly -> end."""
    start, belly, end, at = side
    t = min(max(t, 0.0), 1.0)
    if t <= at:
        u = t / at
        return (start + (belly - start) * math.sin(u * math.pi / 2)) * S
    u = (t - at) / (1 - at)
    return (belly + (end - belly) * (1 - math.cos(u * math.pi / 2))) * S

def outline_points(start: np.ndarray, end: np.ndarray, segment: str, samples: int = 14) -> list[np.ndarray]:
    """Closed outline of a limb: front side going up, rounded end cap, back side coming down."""
    profile = PROFILES[segment]
    vector = end - start
    unit = vector / np.linalg.norm(vector)
    normal = front_normal(unit, segment)
    ts = np.linspace(0, 1, samples)
    front = [start + vector * t + normal * half_width(profile.front, t) for t in ts]
    back = [start + vector * t - normal * half_width(profile.back, t) for t in ts]
    end_cap = _cap(end, unit, normal, half_width(profile.front, 1), half_width(profile.back, 1))
    start_cap = _cap(start, -unit, -normal, half_width(profile.back, 0), half_width(profile.front, 0))
    return front + end_cap + back[::-1] + start_cap

def _cap(centre, forward, normal, r_front, r_back, steps: int = 6) -> list[np.ndarray]:
    """Half-ellipse from the front edge round to the back edge."""
    points = []
    for i in range(1, steps):
        a = math.pi * i / steps
        radius = r_front + (r_back - r_front) * i / steps
        points.append(centre + normal * math.cos(a) * radius + forward * math.sin(a) * radius * 0.9)
    return points

def muscle_points(start: np.ndarray, end: np.ndarray, segment: str, t0: float, t1: float,
                  side: int, depth: float, samples: int = 12) -> tuple[list[np.ndarray], list[list[np.ndarray]]]:
    """Belly outline that follows the limb silhouette, plus two fibre lines inside it."""
    profile = PROFILES[segment]
    vector = end - start
    unit = vector / np.linalg.norm(vector)
    normal = front_normal(unit, segment)
    ts = np.linspace(t0, t1, samples)

    def edge(t: float, sign: int, share: float) -> np.ndarray:
        taper = math.sin(math.pi * (t - t0) / (t1 - t0)) ** 0.6
        width = half_width(profile.front if sign > 0 else profile.back, t)
        return start + vector * t + normal * sign * width * share * max(taper, 0.05)

    if side == 0:
        outer = [edge(t, 1, depth) for t in ts]
        inner = [edge(t, -1, depth) for t in ts]
    else:
        outer = [edge(t, side, depth) for t in ts]
        inner = [start + vector * t + normal * side * half_width(
            profile.front if side > 0 else profile.back, t) * 0.12 for t in ts]
    fibres = []
    for share in (0.35, 0.68):
        fibres.append([i + (o - i) * share for i, o in zip(inner, outer)][1:-1])
    return outer + inner[::-1], fibres

def p3(point) -> np.ndarray:
    return np.array([float(point[0]), float(point[1]), 0.0])

def _polygon(points, fill: str, stroke: str, stroke_width: float, opacity: float = 1.0) -> Polygon:
    return Polygon(*[p3(p) for p in points], color=stroke, stroke_width=stroke_width,
                   fill_color=fill, fill_opacity=opacity)

def _polyline(points, color: str, width: float) -> VMobject:
    line = VMobject().set_points_smoothly([p3(p) for p in points])
    return line.set_stroke(color=color, width=width)

def limb_segments(points: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    neck_top = points["shoulder"] + (points["head"] - points["shoulder"]) * 0.55
    return {
        "shank": (points["ankle"], points["knee"]),
        "thigh": (points["knee"], points["hip"]),
        "torso": (points["hip"], points["shoulder"]),
        "upper_arm": (points["shoulder"], points["elbow"]),
        "forearm": (points["elbow"], points["wrist"]),
        "neck": (points["shoulder"], neck_top),
    }

def body_obstacles(points: dict[str, np.ndarray]) -> list[Capsule]:
    """Capsules approximating the drawn body, used to keep labels off it."""
    capsules = []
    for name, (start, end) in limb_segments(points).items():
        profile = PROFILES[name]
        radius = max(profile.front[1], profile.back[1]) * S
        capsules.append(Capsule(tuple(start), tuple(end), radius))
    head = points["head"]
    capsules.append(Capsule(tuple(head), tuple(head), sk.HEAD_RADIUS * 1.15))
    capsules.append(Capsule(tuple(points["heel"]), tuple(points["toe"]), 0.12 * S))
    for name, (start, end) in free_leg_segments(points).items():
        profile = PROFILES[name]
        capsules.append(Capsule(tuple(start), tuple(end), max(profile.front[1], profile.back[1]) * S))
    if "free_knee" in points:
        capsules.append(Capsule(tuple(points["free_heel"]), tuple(points["free_toe"]), 0.12 * S))
    return capsules

FOOT_SHAPE = [(-0.10, 0.05), (-0.24, -0.06), (-0.26, -0.17), (-0.20, -0.20), (0.42, -0.20),
              (0.44, -0.14), (0.30, -0.10), (0.10, -0.03), (0.06, 0.05)]

def _foot(points, offset=np.zeros(2), ankle_key: str = "ankle", turn_deg: float = 0.0) -> list[np.ndarray]:
    a = points[ankle_key] + offset
    return [a + sk.rotate(np.array(p) * S, turn_deg) for p in FOOT_SHAPE]

def free_leg_segments(points: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Shank and thigh of the non-working leg (single-leg exercises), keyed like limb_segments."""
    if "free_knee" not in points:
        return {}
    return {"shank": (points["free_ankle"], points["free_knee"]),
            "thigh": (points["free_knee"], points["hip"])}

def _head(points, offset=np.zeros(2)) -> list:
    """Skull in profile: an egg shape tilted with the torso and a small nose on the front."""
    centre = points["head"] + offset
    up = (points["head"] - points["shoulder"]) / np.linalg.norm(points["head"] - points["shoulder"])
    face = front_normal(up, "torso")
    r = sk.HEAD_RADIUS
    outline = []
    for i in range(24):
        a = 2 * math.pi * i / 24
        # wider at the back of the skull, narrower at the jaw
        radius_face = r * (1.0 if math.sin(a) >= 0 else 0.92)
        point = centre + face * math.cos(a) * radius_face * 0.95 + up * math.sin(a) * r * 1.12
        outline.append(point)
        if i == 1:  # nose bump just below eye level
            outline.append(centre + face * r * 1.08 + up * r * 0.05)
    return [_polygon(outline, SKIN, OUTLINE, 1.5)]

def build_body(points: dict[str, np.ndarray], activation: dict) -> list:
    """Back-to-front drawing list: far arm and leg, near leg, torso, muscles, head, near arm."""
    segments = limb_segments(points)
    free_segments = free_leg_segments(points)
    far_offset = np.array([-0.06, 0.05]) * S
    shapes = []

    def add_muscle(region, segment_map=segments):
        segment, t0, t1, side, depth = MUSCLES[region]
        start, end = segment_map[segment]
        level = activation.get(region, 0.0)
        outline, fibres = muscle_points(start, end, segment, t0, t1, side, depth)
        stroke = art.PRIMARY_GREEN if level > 0.15 else FIBRE
        shapes.append(_polygon(outline, muscle_color(level), stroke, 1.2))
        for fibre in fibres:
            shapes.append(_polyline(fibre, art.DARK_GREEN if level > 0.5 else FIBRE, 0.8))

    # Far-side arm (lighter, no muscle detail)
    for name in ("upper_arm", "forearm"):
        start, end = segments[name]
        shapes.append(_polygon(outline_points(start + far_offset, end + far_offset, name), SKIN_FAR, OUTLINE_FAR, 1.2))

    if free_segments:
        # Single-leg exercise: the non-working leg is drawn in full, with its own muscles
        turn = float(np.degrees(np.arctan2(*(points["free_knee"] - points["free_ankle"])[::-1]))) - 90.0
        shapes.append(_polygon(_foot(points, ankle_key="free_ankle", turn_deg=turn), SKIN, OUTLINE, 1.5))
        for name in ("shank", "thigh"):
            start, end = free_segments[name]
            shapes.append(_polygon(outline_points(start, end, name), SKIN, OUTLINE, 1.5))
        for region in MUSCLES:
            if MUSCLES[region][0] in ("shank", "thigh"):
                add_muscle(region, free_segments)
    else:
        # Two-legged exercise: the far leg mirrors the near one, lighter and slightly offset
        shapes.append(_polygon(_foot(points, far_offset), SKIN_FAR, OUTLINE_FAR, 1.2))
        for name in ("shank", "thigh"):
            start, end = segments[name]
            shapes.append(_polygon(outline_points(start + far_offset, end + far_offset, name), SKIN_FAR, OUTLINE_FAR, 1.2))

    # Near leg, torso and neck
    shapes.append(_polygon(_foot(points), SKIN, OUTLINE, 1.5))
    for name in ("shank", "thigh", "torso", "neck"):
        start, end = segments[name]
        shapes.append(_polygon(outline_points(start, end, name), SKIN, OUTLINE, 1.5))

    # Glutes: a rounded mass behind the hip joint
    torso_dir = sk.direction(math.degrees(math.atan2(*(points["shoulder"] - points["hip"])[::-1])))
    glute_centre = points["hip"] - front_normal(torso_dir, "torso") * 0.17 * S - torso_dir * 0.02 * S
    glute = Circle(radius=0.26 * S, color=OUTLINE, stroke_width=1.5,
                   fill_color=muscle_color(activation.get(MuscleRegion.GLUTES, 0.0)), fill_opacity=1)
    glute.stretch(1.12, 1).move_to(p3(glute_centre))
    shapes.append(glute)

    for region in MUSCLES:
        if MUSCLES[region][0] in ("shank", "thigh", "torso"):
            add_muscle(region)

    shapes.extend(_head(points))

    # Near arm on top of the torso
    for name in ("upper_arm", "forearm"):
        start, end = segments[name]
        shapes.append(_polygon(outline_points(start, end, name), SKIN, OUTLINE, 1.5))
    for region in MUSCLES:
        if MUSCLES[region][0] in ("upper_arm", "forearm"):
            add_muscle(region)
    hand = Circle(radius=0.09 * S, color=OUTLINE, stroke_width=1.2, fill_color=SKIN, fill_opacity=1)
    shapes.append(hand.move_to(p3(points["wrist"])))
    return shapes
