# render_spec.py
# Renders an animation spec JSON without Gemini or Supabase (free): used to iterate on the Manim scene.
# Usage: python render_spec.py samples/specs/romanian_deadlift_barbell.json [equipment] [title] [--frame SECONDS]
#        python render_spec.py --gallery   (one frame per sample in samples/specs -> media/previews/gallery.png)
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.domain.animation import AnimationSpec
from src.rendering import checks
from src.rendering import skeleton as sk
from src.rendering.runner import MAX_PARALLEL_RENDERS, extract_still, render_animation, render_frame

SAMPLES_DIR = Path("samples/specs")
PREVIEWS_DIR = Path("media/previews")

def load_spec(path: Path) -> AnimationSpec:
    return AnimationSpec.model_validate(json.loads(path.read_text(encoding="utf-8")))

def target_muscles(spec: AnimationSpec) -> list[str]:
    return sorted({item.region.value for kf in spec.keyframes for item in kf.muscle_activation if item.level >= 0.6})

def gallery() -> None:
    """Renders the peak frame of every sample in parallel and tiles them into one contact sheet."""
    from PIL import Image

    entries = json.loads((SAMPLES_DIR / "index.json").read_text(encoding="utf-8"))

    def render(entry: dict) -> Path:
        spec = load_spec(SAMPLES_DIR / entry["file"])
        problems = checks.check_spec_geometry(spec, entry["equipment"])
        if problems:
            print(f"⚠️ {entry['file']}:\n  - " + "\n  - ".join(problems))
        frame = PREVIEWS_DIR / "gallery" / f"{Path(entry['file']).stem}.png"
        render_frame(spec, entry["title"], entry["equipment"], target_muscles(spec), frame,
                     sk.peak_activation_time(spec))
        print(f"✅ {frame}")
        return frame

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_RENDERS) as pool:
        frames = list(pool.map(render, entries))

    tiles = [Image.open(frame).resize((640, 360)) for frame in frames]
    columns = 3
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (640 * columns, 360 * rows), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 640, (index // columns) * 360))
    output = PREVIEWS_DIR / "gallery.png"
    sheet.save(output)
    print(f"Gallery: {output}")

def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python render_spec.py <spec.json> [equipment] [title] [--frame SECONDS] | --gallery")

    args = sys.argv[1:]
    if args[0] == "--gallery":
        gallery()
        return
    frame_at = None
    if "--frame" in args:
        index = args.index("--frame")
        frame_at = float(args[index + 1])
        del args[index:index + 2]
    spec_path = Path(args[0])
    equipment = args[1] if len(args) > 1 else "barbell"
    title = args[2] if len(args) > 2 else spec_path.stem.replace("_", " ")

    spec = load_spec(spec_path)
    problems = checks.check_spec_geometry(spec, equipment)
    if problems:
        print("⚠️ Geometry problems:\n  - " + "\n  - ".join(problems))

    target = target_muscles(spec)
    if frame_at is not None:
        frame = PREVIEWS_DIR / f"{spec_path.stem}_frame_{frame_at:.1f}.png"
        render_frame(spec, title, equipment, target, frame, frame_at)
        print(f"Frame: {frame}")
        return

    video = PREVIEWS_DIR / f"{spec_path.stem}.mp4"
    image = PREVIEWS_DIR / f"{spec_path.stem}.png"
    render_animation(spec, title, equipment, target, video)
    extract_still(video, image, sk.peak_activation_time(spec) + 0.3)
    print(f"Video: {video}\nStill: {image}")

if __name__ == "__main__":
    main()
