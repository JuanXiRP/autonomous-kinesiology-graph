# src/nodes/render_node.py
import logging
from pathlib import Path

from src.domain.state import GraphState
from src.rendering import skeleton as sk
from src.rendering.runner import extract_still, render_animation

def render_exercise_animation(state: GraphState) -> dict:
    """Renders the animation spec with Manim into a video plus a still frame."""
    exercise = state.get("structured_exercise")
    spec = state.get("animation_spec")

    if not exercise:
        return {"validation_errors": ["RenderNode failed: No exercise data."]}
    if not spec:
        return {"validation_errors": ["RenderNode failed: No animation spec."]}

    video_path = Path("media/videos") / f"{exercise.slug}_generative.mp4"
    image_path = Path("media/images") / f"{exercise.slug}.png"
    errors: list[str] = []

    # Never pick up stale output from an earlier run
    video_path.unlink(missing_ok=True)
    image_path.unlink(missing_ok=True)

    try:
        logging.info(f"[RenderNode] Rendering animation for: {exercise.slug}")
        render_animation(spec, exercise.name, exercise.equipment.value, exercise.primary_muscles, video_path)
        exercise.animation_video = str(video_path)
    except Exception as e:
        error_msg = f"Animation render failed: {str(e)}"
        logging.error(error_msg)
        return {"validation_errors": [error_msg]}

    try:
        extract_still(video_path, image_path, sk.peak_activation_time(spec) + 0.3)
        exercise.images = [str(image_path)]
    except Exception as e:
        error_msg = f"Still image extraction failed: {str(e)}"
        logging.error(error_msg)
        errors.append(error_msg)

    return {"structured_exercise": exercise, "validation_errors": errors}
