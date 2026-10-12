# src/rendering/anatomy.py
# Body thickness profiles shared by the drawing (body.py) and the collision checks (checks.py). Pure, no Manim.
import math
from dataclasses import dataclass

import numpy as np

from src.rendering import skeleton as sk

S = sk.SCALE

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

# Front view: limbs seen from the front are symmetric (start, belly, end, belly_position on both sides)
def _symmetric(start: float, belly: float, end: float, at: float) -> Profile:
    return Profile(front=(start, belly, end, at), back=(start, belly, end, at))

FRONT_PROFILES = {
    "thigh": _symmetric(0.2, 0.2, 0.12, 0.3),      # hip -> knee
    "shank": _symmetric(0.11, 0.125, 0.07, 0.3),   # knee -> ankle
    "upper_arm": _symmetric(0.13, 0.12, 0.085, 0.35),
    "forearm": _symmetric(0.085, 0.095, 0.055, 0.3),
    "neck": _symmetric(0.1, 0.09, 0.09, 0.5),
}

def half_width(side: tuple[float, float, float, float], t: float) -> float:
    """Smooth width profile through start -> belly -> end."""
    start, belly, end, at = side
    t = min(max(t, 0.0), 1.0)
    if t <= at:
        u = t / at
        return (start + (belly - start) * math.sin(u * math.pi / 2)) * S
    u = (t - at) / (1 - at)
    return (belly + (end - belly) * (1 - math.cos(u * math.pi / 2))) * S

def front_normal(direction: np.ndarray, segment: str) -> np.ndarray:
    """Unit vector pointing to the front of the body segment (the figure faces right when standing)."""
    if segment in ("upper_arm", "forearm"):  # arm vectors point downwards when hanging
        return np.array([-direction[1], direction[0]])
    return np.array([direction[1], -direction[0]])

def depth_inside(point: np.ndarray, start: np.ndarray, end: np.ndarray, segment: str) -> float:
    """How far `point` sits inside the drawn silhouette of a segment (negative when outside,
    -inf beyond the segment's ends)."""
    vector = end - start
    length = float(np.linalg.norm(vector))
    unit = vector / length
    t = float(np.dot(point - start, unit)) / length
    if not 0.0 <= t <= 1.0:
        return -math.inf
    offset = point - (start + vector * t)
    side = PROFILES[segment].front if float(np.dot(offset, front_normal(unit, segment))) >= 0 else PROFILES[segment].back
    return half_width(side, t) - float(np.linalg.norm(offset))
