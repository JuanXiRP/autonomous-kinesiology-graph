# src/domain/animation.py
from enum import StrEnum
from pydantic import BaseModel, Field, field_validator, model_validator

# This model is also the Gemini structured-output schema for the animation step:
# the Field(description=...) strings are prompt text, so editing one changes what the model generates.

class MuscleRegion(StrEnum):
    HAMSTRINGS = "hamstrings"
    GLUTES = "glutes"
    QUADRICEPS = "quadriceps"
    CALVES = "calves"
    LOWER_BACK = "lower_back"
    UPPER_BACK = "upper_back"
    LATS = "lats"
    CHEST = "chest"
    ABS = "abs"
    SHOULDERS = "shoulders"
    BICEPS = "biceps"
    TRICEPS = "triceps"
    FOREARMS = "forearms"

class ForceKind(StrEnum):
    LOAD = "load"
    MUSCLE_PULL = "muscle_pull"
    GROUND_REACTION = "ground_reaction"

class BodyPoint(StrEnum):
    LOAD = "load"
    WRIST = "wrist"
    ELBOW = "elbow"
    SHOULDER = "shoulder"
    HIP = "hip"
    KNEE = "knee"
    ANKLE = "ankle"

class JointName(StrEnum):
    KNEE = "knee"
    HIP = "hip"
    ELBOW = "elbow"

class PropKind(StrEnum):
    BENCH = "bench"
    FIXED_BAR = "fixed_bar"
    CABLE_ANCHOR = "cable_anchor"

class AnchorKind(StrEnum):
    FEET = "feet"
    HANDS = "hands"

class LoadAttachment(StrEnum):
    HANDS = "hands"
    SHOULDERS = "shoulders"
    NONE = "none"

class MuscleActivation(BaseModel):
    region: MuscleRegion = Field(description="The muscle region being drawn.")
    level: float = Field(description="Tension from 0.0 (relaxed) to 1.0 (maximum contraction or stretch load).")

class Keyframe(BaseModel):
    shank: float = Field(description="Direction in degrees of the vector ankle->knee. 0=pointing right (the direction the figure faces), 90=up, 180=left, 270=down.")
    thigh: float = Field(description="Direction in degrees of the vector knee->hip (same convention).")
    torso: float = Field(description="Direction in degrees of the vector hip->shoulder (same convention).")
    upper_arm: float = Field(description="Direction in degrees of the vector shoulder->elbow (same convention). Hanging straight down is 270.")
    forearm: float = Field(description="Direction in degrees of the vector elbow->wrist (same convention).")
    free_thigh: float | None = Field(default=None, description="ONLY for single-leg exercises (set it in every keyframe), else null. Direction of the vector free-knee->hip of the non-working leg. Hanging straight under the hip = 90; extended straight back, horizontal = 0 (the knee is behind the hip).")
    free_shank: float | None = Field(default=None, description="ONLY for single-leg exercises (set it in every keyframe), else null. Direction of the vector free-ankle->free-knee. Straight leg = same value as free_thigh; bent knee raised in front (knee drive) = free_thigh 200, free_shank 270.")
    duration_s: float = Field(description="Seconds to move from the previous keyframe into this pose (0.3 to 4.0). Ignored for the first keyframe.")
    hold_s: float = Field(default=0.3, description="Seconds to hold this pose (0 to 2).")
    active_step: int = Field(description="Zero-based index of the instruction step highlighted while the figure moves into and holds this pose.")
    muscle_activation: list[MuscleActivation] = Field(description="Tension of the muscle regions involved in this pose. Regions not listed are relaxed.")

class Step(BaseModel):
    title: str = Field(description="Short imperative title, at most 4 words, e.g. 'Hip hinge'.")
    description: str = Field(description="One technical coaching cue, at most 90 characters.")

    @field_validator("title")
    @classmethod
    def title_is_short(cls, value: str) -> str:
        if len(value.split()) > 4 or len(value) > 28:
            raise ValueError(f"Step title too long: '{value}'")
        return value.strip()

    @field_validator("description")
    @classmethod
    def description_is_short(cls, value: str) -> str:
        if len(value) > 90:
            raise ValueError(f"Step description longer than 90 characters: '{value}'")
        return value.strip()

class Prop(BaseModel):
    kind: PropKind = Field(description="bench (drawn under the torso), fixed_bar (overhead bar the hands grab) or cable_anchor (pulley the cable runs to).")
    x: float = Field(default=0.0, description="Only for cable_anchor: horizontal position relative to the feet, in body units (a thigh is about 1.2 units).")
    y: float = Field(default=0.0, description="Only for cable_anchor: height above the floor in body units (a standing body is about 4.5 units tall).")

class ForceVector(BaseModel):
    kind: ForceKind = Field(description="load (gravity/resistance), muscle_pull (what the target muscle does) or ground_reaction.")
    anchor: BodyPoint = Field(description="Body point where the arrow starts. 'load' is the barbell/dumbbell/handle position.")
    angle_deg: float = Field(description="Arrow direction in degrees: 0=right, 90=up, 180=left, 270=down.")
    magnitude: float = Field(description="Relative arrow length from 0.3 (small) to 1.5 (large).")
    label: str = Field(description="At most 12 characters, e.g. 'Load' or 'Hip drive'.")
    active_steps: list[int] = Field(description="Step indices during which the arrow is visible. Empty list means always visible.")

class AnimationSpec(BaseModel):
    subtitle: str = Field(description="Short uppercase-style descriptor, e.g. 'Barbell posterior chain'.")
    anchor: AnchorKind = Field(description="feet: the feet stay planted on the floor (standing, lying, push-up). hands: the hands stay fixed on an overhead bar (pull-up, hanging exercises).")
    load_attachment: LoadAttachment = Field(description="Where the external load is held: hands (barbell, dumbbell, cable handle), shoulders (barbell on the back) or none (bodyweight).")
    balance_over_midfoot: bool = Field(default=False, description="True when the load must travel vertically over the middle of the foot in every pose (deadlifts, Romanian deadlifts, squats, bent-over rows, cleans). False for curls, presses, bench work, cable and bodyweight exercises.")
    reps: int = Field(default=2, description="Repetitions shown, 1 to 3.")
    keyframes: list[Keyframe] = Field(description="3 to 8 poses describing ONE repetition. The last pose must equal the first so the loop closes.")
    steps: list[Step] = Field(description="3 to 5 instruction steps shown in the side panel, in execution order.")
    props: list[Prop] = Field(default_factory=list, description="Static scenery needed by the exercise. Empty for plain standing exercises.")
    force_vectors: list[ForceVector] = Field(description="1 to 4 technical force arrows (load direction, muscle pull, ground reaction).")
    angle_markers: list[JointName] = Field(description="Joints that display a live angle arc with degrees (0 to 3 of knee, hip, elbow).")

    @model_validator(mode="after")
    def check_structure(self) -> "AnimationSpec":
        if not 3 <= len(self.keyframes) <= 8:
            raise ValueError("keyframes must contain between 3 and 8 poses")
        if not 3 <= len(self.steps) <= 5:
            raise ValueError("steps must contain between 3 and 5 items")
        if not 1 <= self.reps <= 3:
            raise ValueError("reps must be between 1 and 3")
        if len(self.angle_markers) > 3:
            raise ValueError("angle_markers allows at most 3 joints")
        if not 1 <= len(self.force_vectors) <= 4:
            raise ValueError("force_vectors must contain between 1 and 4 arrows")

        free_set = [(kf.free_thigh is not None, kf.free_shank is not None) for kf in self.keyframes]
        if any(thigh != shank for thigh, shank in free_set):
            raise ValueError("free_thigh and free_shank must be set together (or both null)")
        if len({thigh for thigh, _ in free_set}) > 1:
            raise ValueError("free_thigh/free_shank must be set in every keyframe or in none")

        first, last = self.keyframes[0], self.keyframes[-1]
        names = ["shank", "thigh", "torso", "upper_arm", "forearm"]
        if self.has_free_leg:
            names += ["free_thigh", "free_shank"]
        for name in names:
            if abs(getattr(first, name) - getattr(last, name)) > 2.0:
                raise ValueError(f"The last keyframe must equal the first one ({name} differs)")

        for index, keyframe in enumerate(self.keyframes):
            if index > 0 and not 0.3 <= keyframe.duration_s <= 4.0:
                raise ValueError(f"keyframe {index}: duration_s must be between 0.3 and 4.0")
            if not 0.0 <= keyframe.hold_s <= 2.0:
                raise ValueError(f"keyframe {index}: hold_s must be between 0 and 2")
            if not 0 <= keyframe.active_step < len(self.steps):
                raise ValueError(f"keyframe {index}: active_step {keyframe.active_step} is not a valid step index")
            for activation in keyframe.muscle_activation:
                if not 0.0 <= activation.level <= 1.0:
                    raise ValueError(f"keyframe {index}: muscle level must be between 0 and 1")
            for angle_name in ("shank", "thigh", "torso", "upper_arm", "forearm"):
                if not -360.0 <= getattr(keyframe, angle_name) <= 720.0:
                    raise ValueError(f"keyframe {index}: {angle_name} angle out of range")

        for force in self.force_vectors:
            if len(force.label) > 12:
                raise ValueError(f"Force label longer than 12 characters: '{force.label}'")
            if not 0.3 <= force.magnitude <= 1.5:
                raise ValueError("Force magnitude must be between 0.3 and 1.5")
            if any(not 0 <= s < len(self.steps) for s in force.active_steps):
                raise ValueError("Force active_steps contains an invalid step index")

        for prop in self.props:
            if prop.kind == PropKind.CABLE_ANCHOR and prop.y <= 0:
                raise ValueError("cable_anchor needs a positive y (height above the floor)")
        return self

    @property
    def has_free_leg(self) -> bool:
        return self.keyframes[0].free_thigh is not None
