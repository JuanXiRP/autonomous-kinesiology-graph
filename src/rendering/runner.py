# src/rendering/runner.py
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from src.domain.animation import AnimationSpec

SCENE_FILE = Path("src/rendering/scene.py")
SCENE_CLASS = "ExerciseScene"
RENDER_TIMEOUT_S = 900

# Manim is CPU-bound: cap simultaneous renders when several variants run concurrently
MAX_PARALLEL_RENDERS = 2
RENDER_SLOTS = threading.BoundedSemaphore(MAX_PARALLEL_RENDERS)

def extract_still(video_path: Path, image_path: Path, at_seconds: float) -> None:
    """Saves the frame closest to `at_seconds` of a video as an image (uses PyAV, no ffmpeg binary needed)."""
    import av

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        container.seek(int(max(at_seconds - 0.5, 0) * 1_000_000))
        best = None
        for frame in container.decode(stream):
            best = frame
            if frame.time is not None and frame.time >= at_seconds:
                break
        if best is None:
            raise RuntimeError("Could not decode any frame from the rendered video")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        best.to_image().save(str(image_path))

def render_animation(spec: AnimationSpec, title: str, equipment: str, target_muscles: list[str],
                     video_path: Path) -> Path:
    """Renders the spec with Manim into `video_path`. Raises on failure."""
    return _render(spec, title, equipment, target_muscles, Path(video_path), still_at=None)

def render_frame(spec: AnimationSpec, title: str, equipment: str, target_muscles: list[str],
                 image_path: Path, at_seconds: float) -> Path:
    """Renders a single frame (fast preview while iterating on the scene)."""
    return _render(spec, title, equipment, target_muscles, Path(image_path), still_at=at_seconds)

def _render(spec, title, equipment, target_muscles, output_path: Path, still_at: float | None) -> Path:
    with RENDER_SLOTS:
        return _render_unlocked(spec, title, equipment, target_muscles, output_path, still_at)

def _render_unlocked(spec, title, equipment, target_muscles, output_path: Path, still_at: float | None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        job_file = tmp_dir / "job.json"
        job_file.write_text(json.dumps({
            "title": title,
            "equipment": equipment,
            "target_muscles": target_muscles,
            "spec": spec.model_dump(mode="json"),
        }), encoding="utf-8")

        env = dict(os.environ)
        env["EXERCISE_RENDER_JOB"] = str(job_file)
        env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
        if still_at is not None:
            env["EXERCISE_STILL_AT"] = str(still_at)

        command = [
            sys.executable, "-m", "manim", "render",
            "-r", "1280,720", "--fps", "30",
            "--media_dir", str(tmp_dir / "media"), "-o", "render",
            "--disable_caching", str(SCENE_FILE), SCENE_CLASS,
        ]
        if still_at is not None:
            command.insert(4, "-s")
        logging.info("🎞️ [Render] Rendering animation with Manim...")
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=RENDER_TIMEOUT_S)
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "")[-600:]
            raise RuntimeError(f"Manim exited with code {result.returncode}: {tail}")

        pattern = "render*.png" if still_at is not None else "render.mp4"
        outputs = list((tmp_dir / "media").rglob(pattern))
        if not outputs or outputs[0].stat().st_size == 0:
            raise RuntimeError("Manim finished but produced no output file")
        shutil.move(str(outputs[0]), str(output_path))

    return output_path
