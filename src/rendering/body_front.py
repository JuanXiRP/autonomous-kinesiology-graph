# src/rendering/body_front.py
# Anatomical figure facing the camera (view = front): torso with shoulders, chest, waist and hips, limbs on
# both sides, and the muscles visible from the front tinted with tension like the side-view figure.
import math

import numpy as np
from manim import Circle, Ellipse

from src.domain.animation import MuscleRegion
from src.rendering import body
from src.rendering import skeleton as sk
from src.rendering.anatomy import FRONT_PROFILES, front_normal
from src.rendering.body import FIBRE, OUTLINE, SKIN, _polygon, _polyline, muscle_color, p3
from src.rendering.layout import Capsule
from src.rendering import palette as art

S = sk.SCALE

# region -> (segment, t0, t1, side, depth) on the limbs; side +1 = towards the body midline, -1 = outer side
LIMB_MUSCLES = {
    MuscleRegion.BICEPS: ("upper_arm", 0.3, 0.92, 0, 0.7),
    MuscleRegion.FOREARMS: ("forearm", 0.05, 0.7, 0, 0.8),
    MuscleRegion.QUADRICEPS: ("thigh", 0.12, 0.9, 0, 0.75),
    MuscleRegion.ADDUCTORS: ("thigh", 0.04, 0.55, 1, 0.85),
    MuscleRegion.ABDUCTORS: ("thigh", 0.0, 0.32, -1, 0.9),
    MuscleRegion.HIP_FLEXORS: ("thigh", 0.0, 0.22, 1, 0.55),
    MuscleRegion.CALVES: ("shank", 0.08, 0.6, 1, 0.7),
}

def axes(points) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors up the torso and towards the screen right of the figure."""
    up = points["shoulder"] - points["hip"]
    up = up / np.linalg.norm(up)
    return up, np.array([up[1], -up[0]])

def limb_segments(points, side: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "upper_arm": (points[f"{side}_shoulder"], points[f"{side}_elbow"]),
        "forearm": (points[f"{side}_elbow"], points[f"{side}_wrist"]),
        "thigh": (points[f"{side}_hip"], points[f"{side}_knee"]),
        "shank": (points[f"{side}_knee"], points[f"{side}_ankle"]),
    }

def torso_outline(points) -> list[np.ndarray]:
    """Shoulders, armpits, waist, hips and crotch, mirrored around the midline."""
    up, right = axes(points)
    neck, pelvis = points["shoulder"], points["hip"]
    width = points["near_shoulder"] - neck
    half = float(np.dot(width, right))

    def side(sign: float) -> list[np.ndarray]:
        r = right * sign
        return [neck + r * (half + 0.02) + up * 0.04,        # top of the shoulder
                neck + r * (half - 0.06) - up * 0.42,        # armpit
                pelvis + r * 0.36 + up * 0.8,                # waist
                pelvis + r * 0.44 + up * 0.12,               # hip
                pelvis + r * 0.12 - up * 0.28]               # inner thigh at the crotch

    right_side, left_side = side(1.0), side(-1.0)
    return [neck + up * 0.12] + right_side + left_side[::-1]

def _torso_region(points, region: MuscleRegion, sign: float) -> list[np.ndarray] | None:
    """Outline of a torso muscle on one side (sign +1 = screen right) or None if not drawn there."""
    up, right = axes(points)
    r = right * sign
    neck, pelvis = points["shoulder"], points["hip"]
    half = float(np.dot(points["near_shoulder"] - neck, right))
    if region == MuscleRegion.CHEST:
        return [neck + r * 0.05 - up * 0.14, neck + r * (half - 0.08) - up * 0.1,
                neck + r * (half - 0.12) - up * 0.42, neck + r * 0.32 - up * 0.62, neck + r * 0.05 - up * 0.6]
    if region == MuscleRegion.ABS:
        return [neck + r * 0.02 - up * 0.66, neck + r * 0.17 - up * 0.66,
                pelvis + r * 0.16 + up * 0.18, pelvis + r * 0.02 + up * 0.14]
    if region == MuscleRegion.OBLIQUES:
        return [neck + r * 0.3 - up * 0.68, neck + r * (half - 0.14) - up * 0.55,
                pelvis + r * 0.4 + up * 0.25, pelvis + r * 0.2 + up * 0.25]
    if region == MuscleRegion.TRAPS:
        return [neck + r * 0.07 + up * 0.28, neck + r * (half - 0.12) + up * 0.04, neck + r * 0.1 - up * 0.04]
    if region == MuscleRegion.LATS:
        return [neck + r * (half - 0.08) - up * 0.42, neck + r * (half - 0.18) - up * 0.48,
                pelvis + r * 0.33 + up * 0.85, pelvis + r * 0.37 + up * 0.85]
    return None

def body_obstacles(points) -> list[Capsule]:
    capsules = []
    for side in ("near", "far"):
        for name, (start, end) in limb_segments(points, side).items():
            capsules.append(Capsule(tuple(start), tuple(end), FRONT_PROFILES[name].front[1] * S))
        capsules.append(Capsule(tuple(points[f"{side}_heel"]), tuple(points[f"{side}_toe"]), 0.1 * S))
    capsules.append(Capsule(tuple(points["hip"]), tuple(points["shoulder"]), 0.42 * S))
    capsules.append(Capsule(tuple(points["head"]), tuple(points["head"]), sk.HEAD_RADIUS * 1.15))
    return capsules

def _foot(points, side: str) -> list[np.ndarray]:
    ankle, heel, toe = points[f"{side}_ankle"], points[f"{side}_heel"], points[f"{side}_toe"]
    across = (toe - heel) / np.linalg.norm(toe - heel)
    return [ankle - across * 0.07, ankle + across * 0.07, toe + np.array([0, 0.05]), toe, heel, heel + np.array([0, 0.05])]

def build_front_body(points: dict[str, np.ndarray], activation: dict) -> list:
    """Drawing list back to front: legs, torso and its muscles, neck and head, arms, hands."""
    shapes = []
    up, right = axes(points)

    def add_limb_muscles(side: str, segment_names: tuple[str, ...]):
        segments = limb_segments(points, side)
        for region, (segment, t0, t1, toward_mid, depth) in LIMB_MUSCLES.items():
            if segment not in segment_names:
                continue
            start, end = segments[segment]
            level = activation.get(region, 0.0)
            # muscle_points' side +1 follows front_normal; turn it into "towards the body midline"
            unit = (end - start) / np.linalg.norm(end - start)
            facing_mid = 1 if float(np.dot(front_normal(unit, segment), points["hip"] - start)) >= 0 else -1
            outline, fibres = body.muscle_points(start, end, segment, t0, t1, toward_mid * facing_mid, depth,
                                                 profile=FRONT_PROFILES[segment])
            shapes.append(_polygon(outline, muscle_color(level), art.PRIMARY_GREEN if level > 0.15 else FIBRE, 1.2))
            for fibre in fibres:
                shapes.append(_polyline(fibre, art.DARK_GREEN if level > 0.5 else FIBRE, 0.8))

    def add_region(outline, level):
        shapes.append(_polygon(outline, muscle_color(level), art.PRIMARY_GREEN if level > 0.15 else FIBRE, 1.2))

    # Legs
    for side in ("far", "near"):
        shapes.append(_polygon(_foot(points, side), SKIN, OUTLINE, 1.5))
        segments = limb_segments(points, side)
        for name in ("shank", "thigh"):
            start, end = segments[name]
            shapes.append(_polygon(body.outline_points(start, end, name, profile=FRONT_PROFILES[name]), SKIN, OUTLINE, 1.5))
        add_limb_muscles(side, ("shank", "thigh"))

    # Torso, its muscles (both sides), neck and head
    shapes.append(_polygon(torso_outline(points), SKIN, OUTLINE, 1.5))
    for region in (MuscleRegion.LATS, MuscleRegion.OBLIQUES, MuscleRegion.ABS, MuscleRegion.CHEST, MuscleRegion.TRAPS):
        for sign in (1.0, -1.0):
            add_region(_torso_region(points, region, sign), activation.get(region, 0.0))
    for k in (0.3, 0.5, 0.7):  # six-pack lines
        a = points["hip"] + (points["shoulder"] - points["hip"]) * k
        shapes.append(_polyline([a - right * 0.15, a, a + right * 0.15], FIBRE, 0.8))
    neck_top = points["shoulder"] + up * 0.3
    shapes.append(_polygon(body.outline_points(points["shoulder"], neck_top, "neck", profile=FRONT_PROFILES["neck"]),
                           SKIN, OUTLINE, 1.5))
    head = Ellipse(width=sk.HEAD_RADIUS * 1.7, height=sk.HEAD_RADIUS * 2.2, color=OUTLINE, stroke_width=1.5,
                   fill_color=SKIN, fill_opacity=1)
    head.rotate(math.atan2(up[1], up[0]) - math.pi / 2).move_to(p3(points["head"]))
    shapes.append(head)

    # Arms over the torso, deltoid caps on the shoulders
    for side in ("far", "near"):
        segments = limb_segments(points, side)
        for name in ("upper_arm", "forearm"):
            start, end = segments[name]
            shapes.append(_polygon(body.outline_points(start, end, name, profile=FRONT_PROFILES[name]), SKIN, OUTLINE, 1.5))
        add_limb_muscles(side, ("upper_arm", "forearm"))
        deltoid = Circle(radius=0.17 * S, color=OUTLINE, stroke_width=1.5,
                         fill_color=muscle_color(activation.get(MuscleRegion.SHOULDERS, 0.0)), fill_opacity=1)
        shapes.append(deltoid.move_to(p3(points[f"{side}_shoulder"] + up * 0.02)))
        hand = Circle(radius=0.085 * S, color=OUTLINE, stroke_width=1.2, fill_color=SKIN, fill_opacity=1)
        shapes.append(hand.move_to(p3(points[f"{side}_wrist"])))
    return shapes
