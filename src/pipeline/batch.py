# src/pipeline/batch.py
# Runs the LangGraph pipeline for several exercise variants at the same time.
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from src.resolution.variants import ExerciseVariant

MAX_PARALLEL_VARIANTS = 3

@dataclass
class VariantResult:
    variant: ExerciseVariant
    status: str  # "created" | "updated" | "no_media" | "failed"
    video_url: str | None = None
    image_urls: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

def initial_state(variant: ExerciseVariant) -> dict:
    return {
        "raw_exercise_name": variant.full_name,
        "variant": variant,
        "structured_exercise": None,
        "animation_spec": None,
        "fallback_spec": None,
        "spec_attempts": 0,
        "validation_errors": [],
        "generation_attempts": 0,
    }

def summarize(variant: ExerciseVariant, final_state: dict, existed: bool) -> VariantResult:
    errors = list(final_state.get("validation_errors") or [])
    exercise = final_state.get("structured_exercise")
    if exercise is None or any(e.startswith(("Database insertion failed", "DBNode failed")) for e in errors):
        return VariantResult(variant, "failed", warnings=errors)

    video = exercise.animation_video if (exercise.animation_video or "").startswith("https://") else None
    images = [url for url in exercise.images if url.startswith("https://")]
    if video is None:
        status = "no_media"
    else:
        status = "updated" if existed else "created"
    return VariantResult(variant, status, video, images, errors)

def run_one(app, variant: ExerciseVariant, existed: bool) -> VariantResult:
    # Name the worker thread after the slug so interleaved log lines stay attributable
    threading.current_thread().name = variant.slug
    try:
        logging.info(f"🚀 Starting pipeline for {variant.full_name}")
        final_state = app.invoke(initial_state(variant))
        return summarize(variant, final_state, existed)
    except Exception as e:  # nodes never raise, but a crash in one variant must not stop the others
        logging.error(f"❌ Pipeline crashed for {variant.slug}: {e}")
        return VariantResult(variant, "failed", warnings=[f"Pipeline crashed: {e}"])

def run_variants(variants: list[ExerciseVariant], existing: set[str], app=None,
                 max_workers: int = MAX_PARALLEL_VARIANTS) -> list[VariantResult]:
    """Runs every variant concurrently and returns the results in the input order."""
    if app is None:
        from src.graph import app as compiled_app
        app = compiled_app
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(variants)))) as pool:
        futures = [pool.submit(run_one, app, variant, variant.slug in existing) for variant in variants]
        return [future.result() for future in futures]
