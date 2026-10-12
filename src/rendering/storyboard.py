# src/rendering/storyboard.py
# A few key moments of an animation side by side in one image, for the visual review step.
# One Manim run renders all frames (see runner.render_frames); PyAV decodes them and PIL tiles them.
import tempfile
from pathlib import Path

from src.domain.animation import AnimationSpec
from src.rendering import skeleton as sk
from src.rendering.runner import render_frames

TILE_SIZE = (640, 360)
MAX_TILES = 4

def storyboard_times(spec: AnimationSpec) -> list[float]:
    """The key poses of the first repetition (the closing pose repeats the first one), at most MAX_TILES,
    evenly picked. The dashed load path in each frame shows the trajectory between them."""
    timeline = sk.build_timeline(spec)
    poses = list(range(len(spec.keyframes) - 1))
    if len(poses) > MAX_TILES:
        poses = [poses[round(i * (len(poses) - 1) / (MAX_TILES - 1))] for i in range(MAX_TILES)]
    return [timeline.arrive[i] + spec.keyframes[i].hold_s / 2 for i in poses]

def compose(images: list, size: tuple[int, int] = TILE_SIZE):
    """Tiles PIL images left to right."""
    from PIL import Image

    sheet = Image.new("RGB", (size[0] * len(images), size[1]), "white")
    for index, image in enumerate(images):
        sheet.paste(image.convert("RGB").resize(size), (index * size[0], 0))
    return sheet

def decode_frames(video_path: Path, count: int) -> list:
    import av

    with av.open(str(video_path)) as container:
        frames = [frame.to_image() for frame in container.decode(video=0)]
    if len(frames) < count:
        raise RuntimeError(f"Storyboard video has {len(frames)} frames, expected {count}")
    # Manim may emit an extra frame; take the last frame of each equal share
    share = len(frames) / count
    return [frames[min(len(frames) - 1, int(share * (i + 1)) - 1)] for i in range(count)]

def render_storyboard(spec: AnimationSpec, title: str, equipment: str, target_muscles: list[str],
                      image_path: Path) -> Path:
    """Renders the storyboard image. Raises on failure."""
    times = storyboard_times(spec)
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "storyboard.mp4"
        render_frames(spec, title, equipment, target_muscles, times, video)
        sheet = compose(decode_frames(video, len(times)))
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(image_path)
    return image_path
