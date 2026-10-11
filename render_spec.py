# render_spec.py
# Renders an animation spec JSON without Gemini or Supabase (free): used to iterate on the Manim scene.
# Usage: python render_spec.py samples/romanian_deadlift_spec.json [equipment] [title] [--frame SECONDS]
import json
import sys
from pathlib import Path

from src.domain.animation import AnimationSpec
from src.rendering import skeleton as sk
from src.rendering.runner import extract_still, render_animation, render_frame

def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python render_spec.py <spec.json> [equipment] [title]")

    args = sys.argv[1:]
    frame_at = None
    if "--frame" in args:
        index = args.index("--frame")
        frame_at = float(args[index + 1])
        del args[index:index + 2]
    spec_path = Path(args[0])
    equipment = args[1] if len(args) > 1 else "barbell"
    title = args[2] if len(args) > 2 else spec_path.stem.replace("_", " ")

    spec = AnimationSpec.model_validate(json.loads(spec_path.read_text(encoding="utf-8")))
    problems = sk.check_spec_geometry(spec)
    if problems:
        print("⚠️ Geometry problems:\n  - " + "\n  - ".join(problems))

    target = sorted({item.region.value for kf in spec.keyframes for item in kf.muscle_activation
                     if item.level >= 0.6})
    if frame_at is not None:
        frame = Path("media/previews") / f"{spec_path.stem}_frame_{frame_at:.1f}.png"
        render_frame(spec, title, equipment, target, frame, frame_at)
        print(f"Frame: {frame}")
        return

    video = Path("media/previews") / f"{spec_path.stem}.mp4"
    image = Path("media/previews") / f"{spec_path.stem}.png"
    render_animation(spec, title, equipment, target, video)
    extract_still(video, image, sk.peak_activation_time(spec) + 0.3)
    print(f"Video: {video}\nStill: {image}")

if __name__ == "__main__":
    main()
