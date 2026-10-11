# src/rendering/layout.py
# Pure label placement for the renderer: finds a spot for a text box that does not cover the body,
# the equipment or labels already placed in the same frame. No Manim import, unit-tested.
import math
from dataclasses import dataclass

import numpy as np

# Area of the frame where labels may go (the instruction panel occupies x < -1.6)
BOUNDS = (-1.5, -3.9, 7.0, 3.9)  # min_x, min_y, max_x, max_y

RADII = (0.15, 0.35, 0.6, 0.9, 1.3)
ANGLE_STEPS = 16

@dataclass(frozen=True)
class Capsule:
    """A segment with a thickness; a circle is a capsule whose ends coincide."""
    a: tuple[float, float]
    b: tuple[float, float]
    radius: float

@dataclass(frozen=True)
class Box:
    cx: float
    cy: float
    width: float
    height: float

    @property
    def corners(self) -> list[np.ndarray]:
        hw, hh = self.width / 2, self.height / 2
        return [np.array([self.cx + dx, self.cy + dy]) for dx in (-hw, hw) for dy in (-hh, hh)]

    def intersects(self, other: "Box", margin: float = 0.04) -> bool:
        return (abs(self.cx - other.cx) * 2 < self.width + other.width + 2 * margin and
                abs(self.cy - other.cy) * 2 < self.height + other.height + 2 * margin)

    def inside(self, bounds=BOUNDS) -> bool:
        min_x, min_y, max_x, max_y = bounds
        return (self.cx - self.width / 2 >= min_x and self.cx + self.width / 2 <= max_x and
                self.cy - self.height / 2 >= min_y and self.cy + self.height / 2 <= max_y)

def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    t = 0.0 if denom == 0 else max(0.0, min(1.0, float(np.dot(p - a, ab)) / denom))
    return float(np.linalg.norm(p - (a + t * ab)))

def _point_in_box(p: np.ndarray, box: Box) -> bool:
    return abs(p[0] - box.cx) <= box.width / 2 and abs(p[1] - box.cy) <= box.height / 2

def _closest_point_in_box(p: np.ndarray, box: Box) -> np.ndarray:
    return np.array([
        min(max(p[0], box.cx - box.width / 2), box.cx + box.width / 2),
        min(max(p[1], box.cy - box.height / 2), box.cy + box.height / 2),
    ])

def capsule_box_penetration(capsule: Capsule, box: Box) -> float:
    """How deep the capsule reaches into the box (0 when they do not touch)."""
    a, b = np.array(capsule.a, dtype=float), np.array(capsule.b, dtype=float)
    best = math.inf
    # Distance from the segment to the box: sample the segment densely (cheap and robust)
    for t in np.linspace(0.0, 1.0, 9):
        p = a + (b - a) * t
        if _point_in_box(p, box):
            return capsule.radius + min(box.width, box.height) / 2
        best = min(best, float(np.linalg.norm(p - _closest_point_in_box(p, box))))
    for corner in box.corners:
        best = min(best, _point_segment_distance(corner, a, b))
    return max(0.0, capsule.radius - best)

def overlap_score(box: Box, obstacles: list[Capsule], taken: list[Box]) -> float:
    score = sum(capsule_box_penetration(capsule, box) for capsule in obstacles)
    score += sum(1.0 for other in taken if box.intersects(other))
    if not box.inside():
        score += 5.0
    return score

def candidate_boxes(anchor, width: float, height: float, prefer_angle: float):
    """Yields (key, box) candidates: preferred direction first, then alternating around it."""
    order = [0]
    for i in range(1, ANGLE_STEPS // 2 + 1):
        order.extend([i, -i])
    for r_index, radius in enumerate(RADII):
        for step in order[:ANGLE_STEPS]:
            angle = prefer_angle + step * 2 * math.pi / ANGLE_STEPS
            direction = np.array([math.cos(angle), math.sin(angle)])
            # Push the box out so its nearest edge (not its centre) sits `radius` away
            extent = abs(direction[0]) * width / 2 + abs(direction[1]) * height / 2
            centre = np.array(anchor, dtype=float) + direction * (radius + extent)
            yield (r_index, step), Box(float(centre[0]), float(centre[1]), width, height)

def place_label(anchor, width: float, height: float, obstacles: list[Capsule], taken: list[Box],
                prefer_angle: float = 0.0, previous: tuple | None = None) -> tuple[tuple, Box]:
    """Returns (key, box) for the first candidate free of overlaps. `previous` (the key chosen in the
    last frame) is tried first so labels do not jump around while the figure moves."""
    candidates = list(candidate_boxes(anchor, width, height, prefer_angle))
    if previous is not None:
        for key, box in candidates:
            if key == previous and overlap_score(box, obstacles, taken) == 0:
                return key, box
    best_key, best_box, best_score = None, None, math.inf
    for key, box in candidates:
        score = overlap_score(box, obstacles, taken)
        if score == 0:
            return key, box
        if score < best_score:
            best_key, best_box, best_score = key, box, score
    return best_key, best_box
