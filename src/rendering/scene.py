# src/rendering/scene.py
# Generic Manim scene: renders any AnimationSpec (see src/domain/animation.py).
# Run through src/nodes/render_node.py or render_spec.py; the job file path comes from EXERCISE_RENDER_JOB.
import json
import math
import os
import textwrap
from pathlib import Path

import numpy as np
from manim import (
    Arc, Arrow, Circle, DashedVMobject, DOWN, LEFT, Line, RIGHT, Scene, SurroundingRectangle, Text,
    VGroup, VMobject, ValueTracker, always_redraw, linear, RoundedRectangle, Dot, BOLD, SEMIBOLD,
)

from src.rendering import palette as art
from src.domain.animation import AnchorKind, AnimationSpec, BodyPoint, ForceKind, JointName, PropKind
from src.rendering import body, layout
from src.rendering import skeleton as sk

PANEL_LEFT, PANEL_RIGHT = -6.5, -1.8
PLATE_RADIUS = 0.55

FORCE_COLORS = {
    ForceKind.LOAD: art.EQUIPMENT_COLOR,
    ForceKind.MUSCLE_PULL: art.PRIMARY_GREEN,
    ForceKind.GROUND_REACTION: art.DARK_GREEN,
}

p3 = body.p3

class ExerciseScene(Scene):
    def construct(self):
        job = json.loads(Path(os.environ["EXERCISE_RENDER_JOB"]).read_text(encoding="utf-8"))
        self.job = job
        self.spec = AnimationSpec.model_validate(job["spec"])
        self.equipment = job.get("equipment", "bodyweight")
        self.camera.background_color = art.BACKGROUND
        self.time_tracker = ValueTracker(0.0)
        self._text_cache = {}
        self._label_memory = {}
        self.hand_y = sk.hand_anchor_y(self.spec)

        self.build_panel()
        self.build_scenery()
        self.build_bar_path()

        self.add(always_redraw(self.build_dynamic))
        self.add(always_redraw(self.build_rep_counter))

        still_at = os.environ.get("EXERCISE_STILL_AT")
        if still_at:  # single-frame preview (render_spec.py --frame)
            self.time_tracker.set_value(float(still_at))
            self.update_mobjects(0)
            self.wait(0.1)
            return

        total = sk.total_duration(self.spec)
        self.wait(0.3)
        self.play(self.time_tracker.animate.set_value(total), run_time=total, rate_func=linear)
        self.wait(0.8)

    # ---- helpers -----------------------------------------------------------------------------

    def current_frame(self) -> sk.Frame:
        return sk.frame_at(self.spec, self.time_tracker.get_value())

    def solve(self, pose: sk.Pose) -> dict[str, np.ndarray]:
        return sk.solve(pose, self.spec.anchor, self.hand_y)

    def text(self, content: str, font_size: int, color: str, weight=None) -> Text:
        """Pango layout is slow: build each distinct label once and hand out copies."""
        key = (content, font_size, color, weight)
        if key not in self._text_cache:
            kwargs = {"weight": weight} if weight is not None else {}
            self._text_cache[key] = Text(content, font_size=font_size, color=color, **kwargs)
        return self._text_cache[key].copy()

    def load_position(self, points: dict[str, np.ndarray]) -> np.ndarray | None:
        return sk.load_point(points, self.spec.load_attachment.value)

    # ---- static layers -----------------------------------------------------------------------

    def build_panel(self):
        title_text = self.job.get("title", "EXERCISE").upper()
        title = Text(title_text, font_size=26, weight=BOLD, color=art.DARK_GREEN)
        if title.width > 4.7:
            title.scale_to_fit_width(4.7)
        subtitle = Text(self.spec.subtitle.upper(), font_size=13, weight=SEMIBOLD, color=art.PRIMARY_GREEN)
        if subtitle.width > 4.7:
            subtitle.scale_to_fit_width(4.7)
        title_group = VGroup(title, subtitle).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        title_group.move_to(np.array([PANEL_LEFT + 0.1, 3.0, 0]), aligned_edge=LEFT)

        divider = Line([PANEL_LEFT, 2.4, 0], [PANEL_RIGHT, 2.4, 0], stroke_width=2, color=art.LIGHT_LINE)

        blocks = []
        for index, step in enumerate(self.spec.steps):
            number = Text(f"{index + 1:02d}", font_size=14, weight=BOLD, color=art.PRIMARY_GREEN)
            head_title = Text(step.title.upper(), font_size=15, weight=BOLD, color=art.DARK_GREEN)
            head = VGroup(number, head_title).arrange(RIGHT, buff=0.2)
            description = Text("\n".join(textwrap.wrap(step.description, 40)), font_size=13,
                               color=art.MUTED_TEXT, line_spacing=1.2)
            blocks.append(VGroup(head, description).arrange(DOWN, aligned_edge=LEFT, buff=0.12))
        steps_group = VGroup(*blocks).arrange(DOWN, aligned_edge=LEFT, buff=0.38)
        if steps_group.height > 4.5:
            steps_group.scale_to_fit_height(4.5)
        steps_group.next_to(divider, DOWN, buff=0.4).align_to(title_group, LEFT)
        self.step_blocks = blocks

        for index, block in enumerate(blocks):
            block.add_updater(lambda mob, i=index: mob.set_opacity(
                1.0 if self.current_frame().active_step == i else 0.45))

        highlight = always_redraw(lambda: SurroundingRectangle(
            blocks[self.current_frame().active_step], buff=0.14, corner_radius=0.1,
            color=art.PRIMARY_GREEN, stroke_width=1.5))

        target = ", ".join(m.replace("_", " ") for m in self.job.get("target_muscles", []))
        legend_text = Text(f"Active tension: {target}", font_size=12, weight=BOLD, color=art.DARK_GREEN)
        if legend_text.width > 4.0:
            legend_text.scale_to_fit_width(4.0)
        legend_tag = RoundedRectangle(corner_radius=0.12, width=4.6, height=0.6, color=art.PRIMARY_GREEN,
                                      fill_color="#E8F0EC", fill_opacity=0.8, stroke_width=1.5)
        legend_tag.move_to(np.array([-4.15, -3.1, 0]))
        legend_dot = Dot(radius=0.08, color=art.GLOW_GREEN)
        legend_content = VGroup(legend_dot, legend_text).arrange(RIGHT, buff=0.2).move_to(legend_tag.get_center())

        self.add(title_group, divider, steps_group, highlight, legend_tag, legend_content)

    def build_scenery(self):
        first_points = self.solve(sk.Pose.from_keyframe(self.spec.keyframes[0]))
        if self.spec.anchor == AnchorKind.FEET:
            self.add(Line([0.3, sk.FLOOR_Y, 0], [6.9, sk.FLOOR_Y, 0], stroke_width=3, color=art.LIGHT_LINE))

        for prop in self.spec.props:
            if prop.kind == PropKind.BENCH:
                xs = [first_points[k][0] for k in ("hip", "shoulder", "head")]
                top = min(first_points["hip"][1], first_points["shoulder"][1]) - 0.3
                left, right = min(xs) - 0.4, max(xs) + 0.4
                self.add(Line([left, top, 0], [right, top, 0], stroke_width=14, color="#8A9A93"))
                for x in (left + 0.2, right - 0.2):
                    self.add(Line([x, top, 0], [x, sk.FLOOR_Y, 0], stroke_width=6, color="#8A9A93"))
            elif prop.kind == PropKind.FIXED_BAR:
                x, y = sk.HANDS_ANCHOR_X, self.hand_y
                self.add(Line([x - 1.3, y + 0.12, 0], [x + 1.3, y + 0.12, 0], stroke_width=8, color=art.EQUIPMENT_COLOR))
                for dx in (-1.3, 1.3):
                    self.add(Line([x + dx, y + 0.12, 0], [x + dx, y + 1.0, 0], stroke_width=5, color=art.LIGHT_LINE))
            elif prop.kind == PropKind.CABLE_ANCHOR:
                position = [sk.FEET_ANCHOR_X + prop.x, sk.FLOOR_Y + prop.y, 0]
                self.add(Circle(radius=0.12, color=art.EQUIPMENT_COLOR, fill_color=art.EQUIPMENT_COLOR, fill_opacity=1).move_to(position))

    def build_bar_path(self):
        if self.spec.load_attachment.value == "none":
            return
        rep = sk.build_timeline(self.spec).rep_duration
        samples = []
        for i in range(61):
            frame = sk.frame_at(self.spec, rep * i / 60 - (1e-6 if i == 60 else 0))
            point = self.load_position(self.solve(frame.pose))
            samples.append(p3(point))
        path = VMobject().set_points_as_corners(samples).set_stroke(color="#8FA39A", width=2.5)
        self.add(DashedVMobject(path, num_dashes=45))

    # ---- dynamic layers ----------------------------------------------------------------------

    def build_rep_counter(self):
        rep = sk.build_timeline(self.spec).rep_duration
        current = min(self.spec.reps, int(self.time_tracker.get_value() / rep) + 1)
        label = self.text(f"REP {current}/{self.spec.reps}", 14, art.MUTED_TEXT, BOLD)
        return label.move_to(np.array([6.0, 3.4, 0]), aligned_edge=RIGHT)

    def build_dynamic(self):
        frame = self.current_frame()
        points = self.solve(frame.pose)
        obstacles = body.body_obstacles(points)
        load_shapes, load_obstacle = self.draw_load(points)
        if load_obstacle is not None:
            obstacles.append(load_obstacle)
        taken: list[layout.Box] = []

        group = VGroup()
        group.add(*body.build_body(points, frame.activation))
        group.add(*load_shapes)
        group.add(*self.draw_forces(points, frame, obstacles, taken))
        group.add(*self.draw_angle_markers(points, obstacles, taken))
        return group

    def label(self, key, content: str, anchor, color: str, obstacles, taken, prefer_angle: float, font_size=13):
        """Text on a background chip, placed where it covers neither the body nor other labels."""
        text = self.text(content, font_size, color, BOLD)
        width, height = text.width + 0.16, text.height + 0.12
        chosen, box = layout.place_label(anchor, width, height, obstacles, taken,
                                         prefer_angle, self._label_memory.get(key))
        self._label_memory[key] = chosen
        taken.append(box)
        chip = RoundedRectangle(corner_radius=0.06, width=width, height=height, stroke_width=0,
                                fill_color=art.BACKGROUND, fill_opacity=0.88)
        chip.move_to(np.array([box.cx, box.cy, 0]))
        text.move_to(chip.get_center())
        return [chip, text]

    def draw_load(self, points):
        """Equipment shapes plus the obstacle they occupy (for label placement)."""
        position = self.load_position(points)
        if position is None:
            return [], None
        centre = p3(position)
        dark = art.EQUIPMENT_COLOR
        shapes = []
        radius = 0.12
        if self.equipment == "barbell":
            radius = PLATE_RADIUS
            shapes.append(Circle(radius=radius, color="#2E2E2E", stroke_width=2, fill_color=dark, fill_opacity=1.0).move_to(centre))
            shapes.append(Circle(radius=radius * 0.82, color="#5A5A5A", stroke_width=1.5, fill_opacity=0.0).move_to(centre))
            shapes.append(Circle(radius=radius * 0.3, color="#5A5A5A", stroke_width=1.5, fill_color="#3A3A3A", fill_opacity=1.0).move_to(centre))
            shapes.append(Circle(radius=0.07, color="#1E1E1E", fill_color="#BFC5C2", fill_opacity=1.0).move_to(centre))
        elif self.equipment == "dumbbell":
            radius = 0.24
            shapes.append(Circle(radius=radius, color="#2E2E2E", stroke_width=2, fill_color=dark, fill_opacity=1.0).move_to(centre))
            shapes.append(Circle(radius=0.07, color="#1E1E1E", fill_color="#BFC5C2", fill_opacity=1.0).move_to(centre))
        elif self.equipment == "kettlebell":
            radius = 0.3
            bell = centre + np.array([0, -0.24, 0])
            shapes.append(Circle(radius=0.28, color="#2E2E2E", stroke_width=2, fill_color=dark, fill_opacity=1.0).move_to(bell))
            shapes.append(Arc(radius=0.17, start_angle=0, angle=math.pi, color=dark, stroke_width=6).move_to(centre + np.array([0, 0.02, 0])))
        else:  # machine, cable, band: just the handle
            shapes.append(Circle(radius=radius, color="#2E2E2E", fill_color=dark, fill_opacity=1.0).move_to(centre))
        if self.equipment == "cable":
            for prop in self.spec.props:
                if prop.kind == PropKind.CABLE_ANCHOR:
                    anchor = [sk.FEET_ANCHOR_X + prop.x, sk.FLOOR_Y + prop.y, 0]
                    shapes.insert(0, Line(centre, anchor, stroke_width=3, color="#8FA39A"))
        return shapes, layout.Capsule(tuple(position), tuple(position), radius + 0.04)

    def draw_forces(self, points, frame, obstacles, taken) -> list:
        shapes, pending_labels = [], []
        for index, force in enumerate(self.spec.force_vectors):
            if force.active_steps and frame.active_step not in force.active_steps:
                continue
            if force.anchor == BodyPoint.LOAD:
                origin = self.load_position(points)
                clearance = PLATE_RADIUS if self.equipment == "barbell" else 0.2
                if origin is None:
                    origin, clearance = points["wrist"], 0.1
            else:
                origin, clearance = points[force.anchor.value], 0.25 * sk.SCALE
            direction = sk.direction(force.angle_deg)
            start = p3(origin + direction * clearance)
            end = start + p3(direction * force.magnitude * 0.9)
            color = FORCE_COLORS[force.kind]
            shapes.append(Arrow(start, end, buff=0, stroke_width=6, tip_length=0.22, color=color,
                                max_tip_length_to_length_ratio=0.4))
            # Arrows are obstacles too, so no label (force or angle) is drawn over them
            obstacles.append(layout.Capsule(tuple(start[:2]), tuple(end[:2]), 0.08))
            pending_labels.append((index, force, end, color))
        for index, force, end, color in pending_labels:
            shapes.extend(self.label(("force", index), force.label, end[:2], color, obstacles, taken,
                                     math.radians(force.angle_deg)))
        return shapes

    def draw_angle_markers(self, points, obstacles, taken) -> list:
        shapes = []
        rays = {
            JointName.KNEE: ("ankle", "knee", "hip"),
            JointName.HIP: ("knee", "hip", "shoulder"),
            JointName.ELBOW: ("shoulder", "elbow", "wrist"),
        }
        for joint in self.spec.angle_markers:
            a_name, v_name, b_name = rays[joint]
            a, vertex, b = points[a_name], points[v_name], points[b_name]
            interior = sk.joint_angle(a, vertex, b)
            if interior > 172:
                continue
            a1 = math.atan2(*(a - vertex)[::-1])
            a2 = math.atan2(*(b - vertex)[::-1])
            delta = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
            arc = Arc(radius=0.5, start_angle=a1, angle=delta, arc_center=p3(vertex),
                      color=art.PRIMARY_GREEN, stroke_width=3)
            bisector = a1 + delta / 2
            # The arc sits inside the joint angle; the label goes on the outside of the joint
            shapes.append(arc)
            shapes.extend(self.label(("angle", joint.value), f"{interior:.0f}°", vertex,
                                     art.PRIMARY_GREEN, obstacles, taken, bisector + math.pi, font_size=14))
        return shapes
