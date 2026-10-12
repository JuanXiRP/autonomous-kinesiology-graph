# src/domain/animation.py
from enum import StrEnum
from pydantic import BaseModel, Field, field_validator, model_validator

# This model is also the Gemini structured-output schema for the animation step:
# the Field(description=...) strings are prompt text, so editing one changes what the model generates.

class MovementFamily(StrEnum):
    """Movement pattern of an exercise; picks the worked examples shown to the spec model."""
    HINGE = "hinge"
    SQUAT = "squat"
    LUNGE = "lunge"
    HORIZONTAL_PRESS = "horizontal_press"
    INCLINE_PRESS = "incline_press"
    VERTICAL_PRESS = "vertical_press"
    HORIZONTAL_PULL = "horizontal_pull"
    VERTICAL_PULL = "vertical_pull"
    ELBOW_FLEXION = "elbow_flexion"
    ELBOW_EXTENSION = "elbow_extension"
    HIP_EXTENSION = "hip_extension"
    KNEE_EXTENSION = "knee_extension"
    KNEE_FLEXION = "knee_flexion"
    PUSH_UP_PLANK = "push_up_plank"
    DIP = "dip"
    LATERAL_RAISE = "lateral_raise"
    HIP_ABDUCTION = "hip_abduction"
    CORE_FLEXION = "core_flexion"
    OTHER = "other"

# Movements in the frontal plane: only the front view shows them correctly
FRONT_FAMILIES = {MovementFamily.LATERAL_RAISE, MovementFamily.HIP_ABDUCTION}

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
    TRAPS = "traps"
    OBLIQUES = "obliques"
    ADDUCTORS = "adductors"
    ABDUCTORS = "abductors"
    HIP_FLEXORS = "hip_flexors"

class View(StrEnum):
    SIDE = "side"
    FRONT = "front"

class Bend(StrEnum):
    OUT = "out"
    IN = "in"
    UP = "up"
    DOWN = "down"

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
    SHOULDER = "shoulder"

class LoadAttachment(StrEnum):
    HANDS = "hands"
    SHOULDERS = "shoulders"
    HIPS = "hips"
    FEET = "feet"
    ELBOWS = "elbows"
    KNEES = "knees"
    NONE = "none"

class Posture(StrEnum):
    STANDING = "standing"
    SEATED = "seated"
    SUPINE = "supine"
    PRONE = "prone"
    KNEELING = "kneeling"
    HANGING = "hanging"
    PLANK = "plank"
    BRIDGE = "bridge"

class SupportKind(StrEnum):
    NONE = "none"
    FLAT_BENCH = "flat_bench"
    INCLINE_BENCH = "incline_bench"
    DECLINE_BENCH = "decline_bench"
    SEAT = "seat"
    SEAT_WITH_BACKREST = "seat_with_backrest"
    PREACHER_PAD = "preacher_pad"
    PULL_UP_BAR = "pull_up_bar"
    PARALLEL_BARS = "parallel_bars"
    BOX = "box"

class RigKind(StrEnum):
    FREE = "free"
    MACHINE_LEVER = "machine_lever"
    MACHINE_LINEAR = "machine_linear"
    SMITH = "smith"
    CABLE = "cable"

class CableHeight(StrEnum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"

class CableSide(StrEnum):
    FRONT = "front"
    BACK = "back"
    ABOVE = "above"

class Contact(StrEnum):
    NEAR_HAND = "near_hand"
    FAR_HAND = "far_hand"
    NEAR_FOOT = "near_foot"
    FAR_FOOT = "far_foot"

# Postures each support can be used with
SUPPORT_POSTURES = {
    SupportKind.NONE: set(Posture) - {Posture.HANGING},
    SupportKind.FLAT_BENCH: {Posture.SUPINE, Posture.PRONE, Posture.SEATED, Posture.BRIDGE},
    SupportKind.INCLINE_BENCH: {Posture.SUPINE, Posture.PRONE},
    SupportKind.DECLINE_BENCH: {Posture.SUPINE},
    SupportKind.SEAT: {Posture.SEATED},
    SupportKind.SEAT_WITH_BACKREST: {Posture.SEATED},
    SupportKind.PREACHER_PAD: {Posture.SEATED, Posture.STANDING},
    SupportKind.PULL_UP_BAR: {Posture.HANGING},
    SupportKind.PARALLEL_BARS: {Posture.HANGING},
    SupportKind.BOX: {Posture.STANDING},
}

# Allowed support_angle_deg per support (supports not listed ignore the angle)
SUPPORT_ANGLES = {
    SupportKind.INCLINE_BENCH: (15.0, 80.0),
    SupportKind.DECLINE_BENCH: (-40.0, -5.0),
    SupportKind.SEAT_WITH_BACKREST: (35.0, 95.0),
}

class CableAnchor(BaseModel):
    height: CableHeight = Field(description="Pulley height: low (near the floor), mid (chest height) or high (above the head).")
    side: CableSide = Field(description="Where the cable column stands: front (in front of the figure), back (behind it) or above (pulley right above the hands, e.g. lat pulldown). In front view: front = each hand/foot pulls from the column on its own side, back = from the opposite column, across the body.")

class Setup(BaseModel):
    posture: Posture = Field(description="Body position. standing: the near foot is planted flat on the floor. seated: hips on a seat or on the floor. supine: lying on the back, head to the LEFT. prone: lying face down, head to the RIGHT. kneeling: the near knee on the floor. hanging: the hands hold a bar (pull-ups, dips, hanging raises). plank: toes (or knees) and hands on the floor (push-ups, planks). bridge: upper back on a bench or the floor, feet planted, hips move (hip thrust, glute bridge).")
    support: SupportKind = Field(default=SupportKind.NONE, description="What the body rests on: none, flat_bench, incline_bench, decline_bench, seat (no backrest), seat_with_backrest (machines, seated presses, leg press), preacher_pad, pull_up_bar, parallel_bars (dips) or box (step-ups: the near foot stands on it).")
    support_angle_deg: float = Field(default=0.0, description="Elevation of the bench back or seat backrest above the horizontal: flat bench 0, incline bench 30-45, decline bench -15 to -30, upright machine backrest 80-90, reclined leg press seat 40-50. The torso lies on it: supine/seated torso = 180 - support_angle_deg, prone torso = support_angle_deg.")
    rig: RigKind = Field(default=RigKind.FREE, description="How the load moves: free (free weights or bodyweight), machine_lever (handle or pad on a pivoting lever arm: chest press, leg extension, leg curl), machine_linear (carriage on straight rails: leg press, hack squat), smith (bar on vertical rails) or cable (pulley and cable).")
    cable: CableAnchor | None = Field(default=None, description="Only for rig = cable: where the pulley is. null otherwise.")
    fixed_contacts: list[Contact] = Field(default_factory=list, description="Hands/feet that stay planted at the position they have in the FIRST keyframe while the body moves: e.g. feet on the floor while lying on a bench or seated, hands on the floor in push-ups, the back foot in a split squat or lunge, the free hand on the bench in a one-arm row. Feet within reach of the floor are snapped onto it.")

    @model_validator(mode="after")
    def check_setup(self) -> "Setup":
        if self.posture not in SUPPORT_POSTURES[self.support]:
            allowed = ", ".join(sorted(p.value for p in SUPPORT_POSTURES[self.support]))
            raise ValueError(f"support '{self.support.value}' needs posture {allowed}, not '{self.posture.value}'")
        if self.posture == Posture.HANGING and self.support not in (SupportKind.PULL_UP_BAR, SupportKind.PARALLEL_BARS):
            raise ValueError("posture 'hanging' needs support pull_up_bar or parallel_bars")
        if self.support in SUPPORT_ANGLES:
            low, high = SUPPORT_ANGLES[self.support]
            if not low <= self.support_angle_deg <= high:
                raise ValueError(f"support_angle_deg for {self.support.value} must be between {low:.0f} and {high:.0f}")
        elif self.support == SupportKind.FLAT_BENCH:
            self.support_angle_deg = 0.0
        if self.rig == RigKind.CABLE and self.cable is None:
            raise ValueError("rig 'cable' needs the cable pulley position")
        if self.rig != RigKind.CABLE:
            self.cable = None
        if len(set(self.fixed_contacts)) != len(self.fixed_contacts):
            raise ValueError("fixed_contacts lists a contact twice")
        return self

class Reach(BaseModel):
    angle_deg: float = Field(description="Direction from the limb root (shoulder for an arm, hip for a leg) to the hand or ankle, same angle convention.")
    reach: float = Field(description="Root-to-hand (or root-to-ankle) distance as a fraction of the full limb length: 1.0 = locked straight, 0.5 = deeply bent. Between 0.3 and 1.0.")

class LimbPose(BaseModel):
    proximal: float | None = Field(default=None, description="Angles mode: direction of the upper bone (shoulder->elbow for an arm, hip->knee for a leg).")
    distal: float | None = Field(default=None, description="Angles mode: direction of the lower bone (elbow->wrist for an arm, knee->ankle for a leg).")
    target: Reach | None = Field(default=None, description="Target mode, instead of the angles: where the hand/ankle sits relative to the limb root. The elbow or knee is solved automatically and always bends the anatomical way, and the hand/ankle travels in a straight line between keyframes.")
    bend: Bend | None = Field(default=None, description="Only for view = front with a target: which way the elbow/knee points (out = away from the body midline, in, up, down). null = elbows down, knees out. Ignored in side view.")

    @model_validator(mode="after")
    def one_mode(self) -> "LimbPose":
        angles = self.proximal is not None and self.distal is not None
        partial = (self.proximal is None) != (self.distal is None)
        if partial:
            raise ValueError("a limb in angles mode needs both proximal and distal")
        if angles == (self.target is not None):
            raise ValueError("a limb needs either proximal+distal angles or a target, not both and not neither")
        if self.target is not None and not 0.3 <= self.target.reach <= 1.0:
            raise ValueError("target reach must be between 0.3 and 1.0")
        for value in (self.proximal, self.distal, self.target.angle_deg if self.target else None):
            if value is not None and not -360.0 <= value <= 720.0:
                raise ValueError("limb angle out of range (-360 to 720)")
        return self

    @property
    def uses_target(self) -> bool:
        return self.target is not None

    def differs_from(self, other: "LimbPose", tolerance: float = 2.0) -> bool:
        if self.uses_target != other.uses_target:
            return True
        if self.uses_target:
            return (abs(self.target.angle_deg - other.target.angle_deg) > tolerance
                    or abs(self.target.reach - other.target.reach) > 0.03)
        return abs(self.proximal - other.proximal) > tolerance or abs(self.distal - other.distal) > tolerance

LIMB_NAMES = ("near_arm", "far_arm", "near_leg", "far_leg")

class MuscleActivation(BaseModel):
    region: MuscleRegion = Field(description="The muscle region being drawn.")
    level: float = Field(description="Tension from 0.0 (relaxed) to 1.0 (maximum contraction or stretch load).")

class Keyframe(BaseModel):
    torso: float = Field(description="Direction in degrees of the vector hip->shoulder. 0 = right (the direction the figure faces), 90 = up, 180 = left, 270 = down.")
    near_arm: LimbPose = Field(description="The arm on the camera side (it holds the load). Hanging straight down = proximal 270, distal 270.")
    far_arm: LimbPose | None = Field(default=None, description="The arm on the far side. null = same as near_arm (both arms move together). Set it only for one-arm work.")
    near_leg: LimbPose = Field(description="The leg on the camera side (the stance or front leg). Standing straight = proximal 270, distal 270.")
    far_leg: LimbPose | None = Field(default=None, description="The leg on the far side. null = same as near_leg. Set it for split stances (lunges, split squats: back leg) and single-leg work (the free leg).")
    duration_s: float = Field(description="Seconds to move from the previous keyframe into this pose (0.3 to 4.0). Ignored for the first keyframe.")
    hold_s: float = Field(default=0.3, description="Seconds to hold this pose (0 to 2).")
    active_step: int = Field(description="Zero-based index of the instruction step highlighted while the figure moves into and holds this pose.")
    muscle_activation: list[MuscleActivation] = Field(description="Tension of the muscle regions involved in this pose. Regions not listed are relaxed.")

    def limb(self, name: str) -> LimbPose | None:
        return getattr(self, name)

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

class ForceVector(BaseModel):
    kind: ForceKind = Field(description="load (gravity/resistance), muscle_pull (what the target muscle does) or ground_reaction.")
    anchor: BodyPoint = Field(description="Body point where the arrow starts. 'load' is the barbell/dumbbell/handle/pad position.")
    angle_deg: float = Field(description="Arrow direction in degrees: 0=right, 90=up, 180=left, 270=down.")
    magnitude: float = Field(description="Relative arrow length from 0.3 (small) to 1.5 (large).")
    label: str = Field(description="At most 12 characters, e.g. 'Load' or 'Hip drive'.")
    active_steps: list[int] = Field(description="Step indices during which the arrow is visible. Empty list means always visible.")

# What the front view can show: upright work in the frontal plane (raises, abduction, flyes)
FRONT_POSTURES = {Posture.STANDING, Posture.SEATED, Posture.KNEELING}
FRONT_SUPPORTS = {SupportKind.NONE, SupportKind.SEAT, SupportKind.SEAT_WITH_BACKREST}
FRONT_RIGS = {RigKind.FREE, RigKind.CABLE, RigKind.MACHINE_LEVER}

class AnimationSpec(BaseModel):
    subtitle: str = Field(description="Short uppercase-style descriptor, e.g. 'Barbell posterior chain'.")
    view: View = Field(default=View.SIDE, description="side (default: almost every exercise) or front (movements in the frontal plane: lateral raises, cable crossovers/flyes, hip abduction/adduction). In front view the figure faces the camera, angles are screen directions (0 = screen right, 90 = up), near_* limbs are on the screen RIGHT and a null far_* limb mirrors the near one across the body midline (a lateral raise only needs the near arm going from 270 towards 0).")
    setup: Setup = Field(description="Body position, what it rests on, how the load is guided and which hands/feet stay planted.")
    load_attachment: LoadAttachment = Field(description="Where the external load is: hands (barbell, dumbbell, handle), shoulders (barbell on the back), hips (barbell across the hips in a hip thrust), feet (leg press platform, leg extension/curl pad), elbows (machine pads on the upper arms: lateral raise machine), knees (machine pads on the knees: hip abduction/adduction machine) or none (bodyweight).")
    balance_over_midfoot: bool = Field(default=False, description="True when the load must travel vertically over the middle of the foot in every pose (standing deadlifts, Romanian deadlifts, squats, bent-over rows, cleans). False for everything else.")
    reps: int = Field(default=2, description="Repetitions shown, 1 to 3.")
    keyframes: list[Keyframe] = Field(description="3 to 8 poses describing ONE repetition. The last pose must equal the first so the loop closes.")
    steps: list[Step] = Field(description="3 to 5 instruction steps shown in the side panel, in execution order.")
    force_vectors: list[ForceVector] = Field(description="1 to 4 technical force arrows (load direction, muscle pull, ground reaction).")
    angle_markers: list[JointName] = Field(description="Joints that display a live angle arc with degrees (0 to 3 of knee, hip, elbow, shoulder).")

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

        for name in ("far_arm", "far_leg"):
            present = {keyframe.limb(name) is not None for keyframe in self.keyframes}
            if len(present) > 1:
                raise ValueError(f"{name} must be set in every keyframe or in none")

        first, last = self.keyframes[0], self.keyframes[-1]
        if abs(first.torso - last.torso) > 2.0:
            raise ValueError("The last keyframe must equal the first one (torso differs)")
        for name in LIMB_NAMES:
            a, b = first.limb(name), last.limb(name)
            if a is not None and b is not None and a.differs_from(b):
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
            if not -360.0 <= keyframe.torso <= 720.0:
                raise ValueError(f"keyframe {index}: torso angle out of range")

        if self.view == View.FRONT:
            if self.setup.posture not in FRONT_POSTURES:
                raise ValueError("view 'front' supports standing, seated or kneeling postures only")
            if self.setup.support not in FRONT_SUPPORTS:
                raise ValueError("view 'front' supports no support, a seat or a seat with backrest only")
            if self.setup.rig not in FRONT_RIGS:
                raise ValueError("view 'front' supports free weights, cables and lever machines only")

        for force in self.force_vectors:
            if len(force.label) > 12:
                raise ValueError(f"Force label longer than 12 characters: '{force.label}'")
            if not 0.3 <= force.magnitude <= 1.5:
                raise ValueError("Force magnitude must be between 0.3 and 1.5")
            if any(not 0 <= s < len(self.steps) for s in force.active_steps):
                raise ValueError("Force active_steps contains an invalid step index")
        return self

    def has_independent(self, name: str) -> bool:
        """True when the far arm/leg has its own poses instead of mirroring the near one."""
        return self.keyframes[0].limb(name) is not None
