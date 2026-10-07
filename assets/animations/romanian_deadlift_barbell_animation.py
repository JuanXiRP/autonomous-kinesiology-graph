from manim import *

class ExerciseAnimation(Scene):
    def construct(self):
        # Set background color
        self.camera.background_color = "#F8FAF9"

        # ----------------------------------------------------
        # UI & TEXT PANEL (LEFT SIDE) - PERMANENT
        # ----------------------------------------------------
        title = Text("ROMANIAN DEADLIFT", font_size=28, weight=BOLD, color="#1B4332")
        subtitle = Text("Barbell Posterior Chain", font_size=16, weight=MEDIUM, color="#2D6A4F")
        header_group = VGroup(title, subtitle).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        header_group.to_corner(UL, buff=0.8)

        # Execution steps box
        step1_title = Text("1. Hip Hinge", font_size=16, weight=BOLD, color="#1B4332")
        step1_desc = Text("Push pelvis backward, soft knees,\nmaintain a neutral flat spine.", font_size=13, color="#1B4332", line_spacing=1.2)
        s1 = VGroup(step1_title, step1_desc).arrange(DOWN, aligned_edge=LEFT, buff=0.08)

        step2_title = Text("2. Vertical Bar Path", font_size=16, weight=BOLD, color="#1B4332")
        step2_desc = Text("Keep the barbell skimming close\nto thighs and shins.", font_size=13, color="#1B4332", line_spacing=1.2)
        s2 = VGroup(step2_title, step2_desc).arrange(DOWN, aligned_edge=LEFT, buff=0.08)

        step3_title = Text("3. Hamstring Stretch", font_size=16, weight=BOLD, color="#1B4332")
        step3_desc = Text("Descend to mid-shin level without\nrounding lumbar spine.", font_size=13, color="#1B4332", line_spacing=1.2)
        s3 = VGroup(step3_title, step3_desc).arrange(DOWN, aligned_edge=LEFT, buff=0.08)

        step4_title = Text("4. Hip Drive", font_size=16, weight=BOLD, color="#1B4332")
        step4_desc = Text("Drive hips forward, squeezing glutes\nand hamstrings to lockout.", font_size=13, color="#1B4332", line_spacing=1.2)
        s4 = VGroup(step4_title, step4_desc).arrange(DOWN, aligned_edge=LEFT, buff=0.08)

        steps_vgroup = VGroup(s1, s2, s3, s4).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        steps_vgroup.next_to(header_group, DOWN, buff=0.45, aligned_edge=LEFT)

        # Muscle engagement legend
        legend_title = Text("TARGET MUSCLES:", font_size=12, weight=BOLD, color="#1B4332")
        ham_badge = Dot(radius=0.08, color="#2D6A4F")
        ham_lbl = Text("Hamstrings & Glutes (Active)", font_size=12, color="#2D6A4F")
        legend_row = VGroup(ham_badge, ham_lbl).arrange(RIGHT, buff=0.15)
        legend_box = VGroup(legend_title, legend_row).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        legend_box.next_to(steps_vgroup, DOWN, buff=0.4, aligned_edge=LEFT)

        left_panel = VGroup(header_group, steps_vgroup, legend_box)

        # Step highlight indicators
        step_indicators = [
            SurroundingRectangle(s1, color="#2D6A4F", buff=0.08, stroke_width=1.5, corner_radius=0.1),
            SurroundingRectangle(s2, color="#2D6A4F", buff=0.08, stroke_width=1.5, corner_radius=0.1),
            SurroundingRectangle(s3, color="#2D6A4F", buff=0.08, stroke_width=1.5, corner_radius=0.1),
            SurroundingRectangle(s4, color="#2D6A4F", buff=0.08, stroke_width=1.5, corner_radius=0.1),
        ]

        # ----------------------------------------------------
        # EXERCISE FIGURE & RIG SETUP (RIGHT SIDE)
        # ----------------------------------------------------
        # Floor
        floor = Line(start=[0.5, -2.8, 0], end=[6.5, -2.8, 0], stroke_width=4, color="#1B4332")

        # Barbell vertical path reference guide (dashed line)
        bar_guide = DashedLine(start=[4.3, 0.4, 0], end=[4.3, -2.2, 0], stroke_width=2, color="#444444", dash_length=0.1)

        # Kinematic Tracker: 0.0 = Upright, 1.0 = Fully Hinged
        t_tracker = ValueTracker(0.0)

        # Anchors and Kinematics function
        def get_pose():
            t = t_tracker.get_value()

            # Feet fixed on floor
            ankle = np.array([3.7, -2.5, 0.0])
            foot_heel = np.array([3.3, -2.75, 0.0])
            foot_toe = np.array([4.4, -2.75, 0.0])

            # Knee has soft bend and shifts very slightly back during hinge
            knee = np.array([
                interpolate(3.85, 3.55, t),
                interpolate(-1.25, -1.25, t),
                0.0
            ])

            # Pelvis / Hips push significantly backwards
            hip = np.array([
                interpolate(3.8, 2.1, t),
                interpolate(0.0, -0.15, t),
                0.0
            ])

            # Torso: length approx 1.7 units. Neutral flat spine rotating forward
            # Standing angle: ~85 degrees. Hinged angle: ~12 degrees above horizontal
            torso_length = 1.75
            angle = interpolate(85 * DEGREES, 10 * DEGREES, t)
            shoulder = hip + np.array([
                torso_length * np.cos(angle),
                torso_length * np.sin(angle),
                0.0
            ])

            # Head continues neutral cervical spine line
            neck_vec = np.array([np.cos(angle), np.sin(angle), 0.0])
            head_center = shoulder + 0.38 * neck_vec

            # Arm hangs naturally vertical from shoulder holding the barbell
            arm_length = 1.6
            hand = shoulder + np.array([0.0, -arm_length, 0.0])

            return ankle, foot_heel, foot_toe, knee, hip, shoulder, head_center, hand

        # Dynamic figure creation
        def make_figure():
            ankle, foot_heel, foot_toe, knee, hip, shoulder, head_center, hand = get_pose()
            t = t_tracker.get_value()

            figure = VGroup()

            # Foot / Shoe
            foot = Polygon(foot_heel, foot_toe, [foot_toe[0], -2.6, 0], [ankle[0], -2.5, 0],
                           fill_color="#1B4332", fill_opacity=1.0, stroke_width=0)
            figure.add(foot)

            # Calf / Lower leg (Solid structure)
            shin = Line(ankle, knee, stroke_width=14, stroke_color="#1B4332")
            figure.add(shin)

            # Knee Joint
            knee_joint = Dot(knee, radius=0.12, color="#1B4332")
            figure.add(knee_joint)

            # Thigh: Hamstring prominence (Primary Green highlight) + Quad
            quad_thigh = Line(knee, hip, stroke_width=16, stroke_color="#1B4332")
            # Hamstring active muscular band (posterior side of femur)
            ham_offset = np.array([-0.09, 0.03, 0.0])
            hamstring_layer = Line(knee + ham_offset, hip + ham_offset,
                                   stroke_width=12 + int(6 * t),
                                   stroke_color="#2D6A4F")
            figure.add(quad_thigh, hamstring_layer)

            # Hip & Muscular Gluteus (Highlighted active Primary Green)
            glute_center = hip + np.array([-0.22, -0.05, 0.0])
            glute = Ellipse(width=0.48 + 0.1 * t, height=0.42,
                            fill_color="#2D6A4F", fill_opacity=1.0,
                            stroke_color="#1B4332", stroke_width=2)
            glute.move_to(glute_center)
            figure.add(glute)

            # Robust Torso / Spinal Column (Solid, flat neutral spine)
            torso_line = Line(hip, shoulder, stroke_width=20, stroke_color="#1B4332")
            # Spinal erector / back contour
            back_contour = Line(hip + np.array([-0.05, 0.1, 0.0]),
                                shoulder + np.array([-0.05, 0.1, 0.0]),
                                stroke_width=10, stroke_color="#2D6A4F")
            figure.add(torso_line, back_contour)

            # Shoulder Joint
            shoulder_joint = Dot(shoulder, radius=0.14, color="#1B4332")
            figure.add(shoulder_joint)

            # Neck & Head
            head = Circle(radius=0.22, fill_color="#1B4332", fill_opacity=1.0, stroke_color="#1B4332")
            head.move_to(head_center)
            figure.add(head)

            # Arm (hanging vertically)
            arm = Line(shoulder, hand, stroke_width=12, stroke_color="#1B4332")
            hand_dot = Dot(hand, radius=0.12, color="#1B4332")
            figure.add(arm, hand_dot)

            # Barbell: Side profile view (Plate + Bar collar)
            bar_shaft = Dot(hand, radius=0.08, color="#111111")
            plate = Circle(radius=0.42, stroke_width=6, stroke_color="#444444",
                           fill_color="#555555", fill_opacity=0.35)
            plate.move_to(hand)
            plate_rim = Circle(radius=0.32, stroke_width=2, stroke_color="#222222")
            plate_rim.move_to(hand)
            barbell_group = VGroup(plate, plate_rim, bar_shaft)
            figure.add(barbell_group)

            return figure

        figure_mobject = always_redraw(make_figure)

        # ----------------------------------------------------
        # PRESENTATION & ANIMATION SEQUENCE
        # ----------------------------------------------------
        # 1. Entrance: Left text panel & basic scene setup
        self.play(
            FadeIn(left_panel, shift=RIGHT * 0.2),
            Create(floor),
            Create(bar_guide),
            run_time=1.2
        )
        self.play(FadeIn(figure_mobject), run_time=0.8)
        self.wait(0.5)

        # Repetition Loop (2 Clean, Educational Repetitions)
        indicator = step_indicators[0].copy()
        self.play(Create(indicator), run_time=0.4)

        for rep in range(2):
            # Phase 1 & 2: Descent (Hip Hinge + Bar Descent along shins)
            self.play(
                Transform(indicator, step_indicators[0]),
                run_time=0.4
            )
            self.play(
                Transform(indicator, step_indicators[1]),
                run_time=0.4
            )
            self.play(
                t_tracker.animate.set_value(1.0),
                run_time=2.2,
                rate_func=smooth
            )

            # Phase 3: Pause at stretch depth (mid-shin stretch, flat back)
            self.play(
                Transform(indicator, step_indicators[2]),
                run_time=0.4
            )
            self.wait(0.6)

            # Phase 4: Ascend / Drive hips forward to standing lockout
            self.play(
                Transform(indicator, step_indicators[3]),
                run_time=0.4
            )
            self.play(
                t_tracker.animate.set_value(0.0),
                run_time=1.8,
                rate_func=smooth
            )
            self.wait(0.5)

        # Final hold showing completion
        self.play(FadeOut(indicator), run_time=0.4)
        self.wait(1.0)