# src/nodes/storage_node.py
import logging
import mimetypes
from src.domain.state import GraphState
from src.infrastructure.db.storage_client import (
    SupabaseStorageClient,
    VIDEO_BUCKET,
    IMAGE_BUCKET,
)

def upload_exercise_media(state: GraphState) -> dict:
    """
    Uploads the generated video and images to Supabase Storage and replaces
    the local paths with public URLs. A local path never leaves this node.
    """
    exercise = state.get("structured_exercise")

    if not exercise:
        return {"validation_errors": ["StorageNode failed: No exercise data."]}

    errors: list[str] = []

    try:
        storage = SupabaseStorageClient()
    except Exception as e:
        error_msg = f"StorageNode failed: could not create storage client: {str(e)}"
        logging.error(error_msg)
        exercise.images = []
        exercise.animation_video = None
        return {"structured_exercise": exercise, "validation_errors": [error_msg]}

    if exercise.animation_video:
        try:
            exercise.animation_video = storage.upload_file(
                exercise.animation_video, exercise.slug, VIDEO_BUCKET, "video/mp4"
            )
        except Exception as e:
            error_msg = f"StorageNode video upload failed: {str(e)}"
            logging.error(error_msg)
            exercise.animation_video = None
            errors.append(error_msg)

    uploaded_images: list[str] = []
    for image_path in exercise.images:
        try:
            uploaded_images.append(
                storage.upload_file(
                    image_path, exercise.slug, IMAGE_BUCKET,
                    mimetypes.guess_type(image_path)[0] or "image/jpeg",
                )
            )
        except Exception as e:
            error_msg = f"StorageNode image upload failed for {image_path}: {str(e)}"
            logging.error(error_msg)
            errors.append(error_msg)
    exercise.images = uploaded_images

    return {"structured_exercise": exercise, "validation_errors": errors}
