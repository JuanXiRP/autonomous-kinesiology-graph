# src/nodes/spec_node.py
import json
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.domain.animation import FRONT_FAMILIES, AnimationSpec, MovementFamily, View
from src.domain.state import GraphState
from src.rendering import checks

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "samples" / "specs"
DEFAULT_FAMILY = "hinge"
MAX_EXAMPLES = 2
FEEDBACK_PREFIXES = ("Spec generation failed", "Visual review rejected")

SYSTEM_PROMPT = """You are a biomechanics animator. You describe ONE repetition of a gym exercise as a short list of body poses (keyframes) that a deterministic renderer animates in side view (or front view for frontal-plane movements).

ANGLE CONVENTION (strict):
- The figure faces RIGHT when standing. Angles are in degrees and give the direction of a vector: 0 = right (forward), 90 = up, 180 = left (backwards), 270 = down.
- torso = hip->shoulder. Arms: proximal = shoulder->elbow, distal = elbow->wrist. Legs: proximal = hip->knee, distal = knee->ankle.
- Standing tall, arms hanging: torso 90, arms {proximal 270, distal 270}, legs {proximal 270, distal 270}.
- A squat or hinge bends the knee forward: leg proximal below 270 towards 300-350 (knee in front of the hip), distal around 240-265 (ankle behind the knee).

VIEW:
- view = side for almost everything (squats, hinges, presses, rows, pulls, curls, extensions, lunges, core).
- view = front ONLY for movements in the frontal plane: lateral raises, cable crossovers and flyes seen from the front, hip abduction/adduction. The figure faces the camera, angles are screen directions, near_* limbs are on the screen right and a null far_* limb mirrors the near one across the body midline (a lateral raise moves the near arm from proximal 276 towards 358). Single-leg front work: the near leg moves, the far leg stands. Postures standing, seated or kneeling; rig free, cable or machine_lever (lateral raise machine with load_attachment elbows, hip abduction/adduction machine with load_attachment knees). Seated in front view the thighs point at the camera: give the leg proximal 270 for knees together and swing it outwards (e.g. 310) to spread the knees. In front view a cable on side 'front' pulls each hand from the column on its own side, 'back' from the opposite column (across the body). An optional bend (out, in, up, down) sets which way a target-mode elbow/knee points.

SETUP (choose it first, it decides where the body is anchored and which scenery is drawn):
- posture: standing (the near foot is planted), seated, supine (on the back, head LEFT: torso about 180 on a flat bench), prone (face down, head RIGHT: torso about 0), kneeling, hanging (hands on a bar), plank (toes and hands on the floor), bridge (upper back on a bench or floor, hips move).
- support + support_angle_deg: what the body rests on. For benches and backrests the torso lies on the pad, so supine/seated torso = 180 - support_angle_deg (flat bench 180, incline 35 -> 145, upright backrest 90 -> 90) and prone torso = support_angle_deg.
- rig: free weights/bodyweight = free; selectorized or plate-loaded machines with a pivoting arm = machine_lever; sleds and carriages on rails (leg press, hack squat) = machine_linear; smith = smith; cables = cable (+ cable pulley height and side).
- fixed_contacts: hands/feet that stay planted where the first keyframe puts them: feet on the floor when seated or lying on a bench, hands on the floor in push-ups, the back foot of a lunge or split squat, the free hand on a bench in one-arm rows. The renderer keeps them glued and solves the knees/elbows.

LIMBS (each limb is EITHER angles OR a target):
- angles {proximal, distal}: use when a joint must stay fixed or rotate on its own: curls and triceps extensions (upper arm fixed, forearm rotates), raises, hanging arms in deadlifts, and the legs of most standing exercises.
- target {angle_deg, reach}: use when the hand/foot path matters: presses, rows, pulldowns, leg press. angle_deg is the direction from the shoulder (arm) or hip (leg) to the hand/ankle; reach is that distance as a fraction of the full limb length (1.0 = locked straight, 0.3 = fully bent). The elbow/knee is solved automatically and always bends the anatomical way, and the hand travels in a straight line between keyframes.
- far_arm / far_leg: null means "same as the near one" (both limbs move together). Set them in EVERY keyframe only for one-arm work, split stances (back leg) and single-leg work (the free leg).
- Angles are interpolated numerically between keyframes, so keep consecutive values on the same side of the turn: a curl goes from distal 270 to 410 (not to 50), a leg extension from distal 270 to 360 (not to 0).
- The load sits at the near hand (load_attachment = hands), on the shoulders, across the hips or at the feet (leg press platform, leg extension/curl pad). With the bar on the shoulders or the hips the hands are placed on the bar automatically: give the arms any pose (e.g. proximal 270, distal 270).

RULES:
- keyframes describe ONE repetition; the LAST keyframe must equal the FIRST so the loop closes. Use 4 to 6 keyframes: start, descent/phase, peak, ascent, finish.
- Respect human joint limits: knee and elbow flexion up to about 150 degrees, hip flexion up to about 130 degrees, knees and elbows never bend backwards.
- A barbell never passes through the body, the head or the bench: it stops on the chest surface (in supine presses keep reach about 0.33 or more at the chest touch).
- balance_over_midfoot: true for standing deadlifts, Romanian deadlifts, squats, bent-over rows and cleans; then the load stays vertically over the middle of the near foot in every keyframe.
- muscle_activation: only the regions the exercise really loads, higher tension at the peak of effort. Map the muscles of the exercise to the available regions (hamstrings, glutes, quadriceps, calves, lower_back, upper_back, lats, chest, abs, shoulders, biceps, triceps, forearms, traps, obliques, adductors, abductors, hip_flexors).
- steps: 3 to 5 coaching steps in execution order, technical language, short. Each keyframe's active_step says which step is highlighted while the figure moves into that pose.
- force_vectors: 1 to 4 arrows with a clear teaching purpose (load direction, what the target muscle drives, ground reaction when relevant). Label at most 12 characters. Avoid arrows that overlap each other.
- angle_markers: the 1-3 joints whose angle matters for the technique (knee, hip, elbow, shoulder).

Return only the structured data.

WORKED EXAMPLES (same schema), as JSON:
{examples}
"""

def _index() -> list[dict]:
    return json.loads((SAMPLES_DIR / "index.json").read_text(encoding="utf-8"))

def select_examples(family: str | None, equipment: str | None = None) -> list[dict]:
    """The hand-made sample specs closest to the exercise: same movement family first (same equipment
    preferred), padded with the default family so the model always sees at least one complete example."""
    entries = _index()
    family = family or DEFAULT_FAMILY
    same = sorted((e for e in entries if e["family"] == family), key=lambda e: e["equipment"] != equipment)
    chosen = same[:MAX_EXAMPLES]
    if not chosen:
        chosen = [e for e in entries if e["family"] == DEFAULT_FAMILY][:1]
    return chosen

def render_examples(entries: list[dict]) -> str:
    blocks = []
    for entry in entries:
        spec = json.loads((SAMPLES_DIR / entry["file"]).read_text(encoding="utf-8"))
        blocks.append(f"{entry['title']}:\n{json.dumps(spec, separators=(',', ':'))}")
    return "\n\n".join(blocks)

def build_system_prompt(family: str | None, equipment: str | None) -> str:
    return SYSTEM_PROMPT.replace("{examples}", render_examples(select_examples(family, equipment)))

def generate_animation_spec(state: GraphState) -> dict:
    exercise = state.get("structured_exercise")
    attempts = state.get("spec_attempts", 0)

    if not exercise:
        return {"validation_errors": ["SpecNode failed: No exercise data."], "spec_attempts": attempts + 1}

    previous = [e for e in state.get("validation_errors", []) if e.startswith(FEEDBACK_PREFIXES)][-2:]
    variant = state.get("variant")
    family = getattr(variant, "movement_family", None)
    family = getattr(family, "value", family)

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.2, max_retries=2)
        structured_llm = llm.with_structured_output(AnimationSpec)

        request = (
            f"Exercise: {exercise.name}\n"
            f"Equipment: {exercise.equipment.value}\n"
            f"Primary muscles: {', '.join(exercise.primary_muscles)}\n"
            f"Secondary muscles: {', '.join(exercise.secondary_muscles)}\n"
            f"Preparation: {exercise.preparation}\n"
            f"Execution steps: {' | '.join(exercise.execution_steps)}\n"
            f"Common mistakes: {' | '.join(exercise.common_mistakes)}\n"
        )
        if previous:
            request += "\nYour previous attempt was rejected. Fix these problems:\n- " + "\n- ".join(previous)

        spec = structured_llm.invoke([
            SystemMessage(content=build_system_prompt(family, exercise.equipment.value)),
            HumanMessage(content=request),
        ])

        problems = checks.check_spec_geometry(spec, exercise.equipment.value)
        if family in {f.value for f in FRONT_FAMILIES} and spec.view != View.FRONT:
            problems.insert(0, f"{MovementFamily(family).value.replace('_', ' ')} is a frontal-plane movement: "
                               f"use view = front (side view turns it into a different exercise)")
        if problems:
            raise ValueError("; ".join(problems))

        return {"animation_spec": spec, "spec_attempts": attempts + 1}

    except Exception as e:
        error_msg = f"Spec generation failed on attempt {attempts + 1}: {str(e)}"
        logging.error(error_msg)
        return {"validation_errors": [error_msg], "spec_attempts": attempts + 1}
