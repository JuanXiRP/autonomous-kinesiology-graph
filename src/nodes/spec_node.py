# src/nodes/spec_node.py
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.domain.animation import AnimationSpec
from src.domain.state import GraphState
from src.rendering import skeleton as sk

GOLDEN_SPEC_PATH = Path(__file__).resolve().parents[2] / "samples" / "romanian_deadlift_spec.json"

SYSTEM_PROMPT = """You are a biomechanics animator. You describe ONE repetition of an exercise as a short list of body poses (keyframes) that a deterministic renderer animates in side view.

CONVENTIONS (strict):
- The figure always faces RIGHT. Angles are in degrees and describe the direction of a bone vector: 0 = pointing right (forward), 90 = up, 180 = left (backwards), 270 = down.
- shank = ankle->knee, thigh = knee->hip, torso = hip->shoulder, upper_arm = shoulder->elbow, forearm = elbow->wrist.
- Standing upright with arms hanging: shank 90, thigh 90, torso 90, upper_arm 270, forearm 270.
- Hip hinge forward: the torso angle drops towards 0 (leaning forward to the right). A knee bends forward: shank below 90 (tilting right) and thigh above 90.
- Lying supine on a bench with feet on the floor: torso about 180 (head to the left), thigh about 180-ish pointing back along the body, shank up and to the right.
- The feet stay planted (anchor = feet) unless the hands stay on an overhead bar (anchor = hands, e.g. pull-ups).
- The load is held at the wrist (load_attachment = hands) or on the shoulders (barbell back squat). Keep the wrists connected to a plausible load path.

RULES:
- keyframes describe ONE repetition; the LAST keyframe must have the same angles as the FIRST so the loop closes. Use 4 to 6 keyframes: start, descent/phase, peak, ascent, finish.
- balance_over_midfoot: set it to true for deadlifts, Romanian deadlifts, squats, bent-over rows and cleans. Then in EVERY keyframe the load (wrist or shoulder) must sit vertically over the middle of the foot: when the torso leans forward, push the hips back (thigh well above 90) and angle the arms slightly back (upper_arm/forearm around 250-262) so the bar stays over the feet, as in the worked example.
- Single-leg / unilateral lower-body exercises (single-leg RDL, single-leg glute bridge, step-ups...): set free_thigh and free_shank in EVERY keyframe for the non-working leg; leave them null for two-legged exercises. In hinges the free leg extends back in one line with the torso (free_thigh close to the torso angle). The free foot must stay above the floor. Short example (single-leg RDL) as [shank, thigh, torso, upper_arm, forearm, free_thigh, free_shank]: start [90, 90, 90, 270, 270, 75, 55], mid [88, 106, 48, 256, 254, 50, 46], bottom [86, 118, 14, 250, 248, 16, 14].
- Respect human joint limits: knee and elbow flexion up to about 150 degrees, hip flexion up to about 130 degrees; knees never bend backwards.
- muscle_activation: only the regions the exercise really loads, higher tension at the peak of effort. Map the muscles of the exercise to the available regions (hamstrings, glutes, quadriceps, calves, lower_back, upper_back, lats, chest, abs, shoulders, biceps, triceps, forearms).
- steps: 3 to 5 coaching steps in execution order, technical language, short. Each keyframe's active_step says which step is highlighted while the figure moves into that pose.
- force_vectors: 1 to 4 arrows with a clear teaching purpose (load direction, what the target muscle drives, ground reaction when relevant). Label at most 12 characters. Avoid arrows that overlap each other.
- angle_markers: the 1-3 joints whose angle matters for the technique (knee, hip, elbow).
- props: bench for lying/seated pressing, fixed_bar for hanging exercises, cable_anchor (with x, y) for cable exercises. Empty for plain standing work.

Return only the structured data.

WORKED EXAMPLE (Romanian deadlift, barbell), as JSON:
{golden}
"""

def _load_golden() -> str:
    return GOLDEN_SPEC_PATH.read_text(encoding="utf-8")

def generate_animation_spec(state: GraphState) -> dict:
    exercise = state.get("structured_exercise")
    attempts = state.get("spec_attempts", 0)

    if not exercise:
        return {"validation_errors": ["SpecNode failed: No exercise data."], "spec_attempts": attempts + 1}

    previous = [e for e in state.get("validation_errors", []) if e.startswith("Spec generation failed")][-2:]

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
            SystemMessage(content=SYSTEM_PROMPT.replace("{golden}", _load_golden())),
            HumanMessage(content=request),
        ])

        problems = sk.check_spec_geometry(spec)
        if problems:
            raise ValueError("; ".join(problems))

        return {"animation_spec": spec, "spec_attempts": attempts + 1}

    except Exception as e:
        error_msg = f"Spec generation failed on attempt {attempts + 1}: {str(e)}"
        logging.error(error_msg)
        return {"validation_errors": [error_msg], "spec_attempts": attempts + 1}
