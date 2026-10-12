# src/nodes/critic_node.py
# Visual review: renders the key poses of the spec and asks Gemini (multimodal) whether they show the exercise
# correctly. A rejection sends concrete problems back to the spec step; the review never blocks the pipeline.
import base64
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.domain.state import GraphState
from src.rendering.storyboard import render_storyboard

REJECTION_PREFIX = "Visual review rejected"

# This model is the Gemini structured-output schema: the Field descriptions are prompt text.
class VisualReview(BaseModel):
    approved: bool = Field(description="True when a strength coach would accept the frames as a correct demonstration of the exercise.")
    problems: list[str] = Field(default_factory=list, description="Empty when approved. Otherwise 1 to 4 concrete problems, each naming the frame, the body part or equipment, what is wrong and what to change in the spec (torso angle, a limb target or angles, the setup).")

SYSTEM_PROMPT = """You are a strength coach reviewing a technical exercise animation before it is published.
You see the key poses of ONE repetition side by side (left to right in execution order), rendered in strict side view from the animation spec given as JSON. The figure faces right when standing; supine figures have the head on the left.

Judge only whether the movement is a correct, recognisable demonstration of the named exercise:
- body position and support (standing, seated, lying on a bench, hanging...), the bench angle or seat,
- where the load is and the path it follows (dashed line), the equipment and machine,
- joint positions at each key pose (range of motion, depth, lockout), balance and anatomy.
Ignore the drawing style, colours, the side panel text, arrow and label placement, and small proportions. Some arm poses are side-view projections of 3D positions (e.g. elbows tucked in presses): accept them when plausible.
Approve unless something a coach would call wrong is visible. When rejecting, give actionable fixes in terms of the spec fields."""

def _image_part(path: Path) -> dict:
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}

def review_animation_spec(state: GraphState) -> dict:
    exercise = state.get("structured_exercise")
    spec = state.get("animation_spec")
    attempt = state.get("spec_attempts", 0)
    if not exercise or not spec:
        return {"validation_errors": ["CriticNode skipped: no exercise or spec."]}

    try:
        image = render_storyboard(spec, exercise.name, exercise.equipment.value, exercise.primary_muscles,
                                  Path("media/storyboards") / f"{exercise.slug}.png")
        llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.0, max_retries=2)
        review = llm.with_structured_output(VisualReview).invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=[
                {"type": "text", "text": (
                    f"Exercise: {exercise.name}\n"
                    f"Preparation: {exercise.preparation}\n"
                    f"Execution steps: {' | '.join(exercise.execution_steps)}\n"
                    f"Spec: {spec.model_dump_json(exclude={'steps', 'force_vectors'})}"
                )},
                _image_part(image),
            ]),
        ])
    except Exception as e:  # the review is advisory: a failure must not cost the record its video
        logging.warning(f"👁️ [Critic] Visual review skipped: {e}")
        return {"validation_errors": [f"Visual review skipped: {e}"]}

    if review.approved or not review.problems:
        logging.info("👁️ [Critic] Animation approved")
        return {"animation_spec": spec}

    problems = "; ".join(review.problems)
    logging.info(f"👁️ [Critic] Animation rejected: {problems}")
    return {
        "animation_spec": None,
        "fallback_spec": spec,
        "validation_errors": [f"{REJECTION_PREFIX} (attempt {attempt}): {problems}"],
    }
