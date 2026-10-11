# upload_media.py
# Re-uploads the media already rendered locally for exercises stored in the database and points their
# row at the public URLs. Never generates anything (no Gemini, no Manim).
# Usage: python upload_media.py romanian_deadlift_dumbbell [other_slug ...]
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from src.infrastructure.db.storage_client import IMAGE_BUCKET, VIDEO_BUCKET, SupabaseStorageClient

def local_media(slug: str) -> tuple[Path, Path]:
    return Path("media/videos") / f"{slug}_generative.mp4", Path("media/images") / f"{slug}.png"

def repair(slug: str, storage, connect=psycopg.connect) -> dict:
    """Uploads the local video/still of `slug` and updates only the fields that uploaded."""
    db_url = os.environ.get("DATABASE_URL")
    with connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM exercises WHERE slug = %s", (slug,))
            if cur.fetchone() is None:
                raise LookupError(f"'{slug}' is not in the database; generate it with main.py first")

        video_path, image_path = local_media(slug)
        updates, errors = {}, []
        if video_path.exists():
            try:
                updates["animation_video"] = storage.upload_file(str(video_path), slug, VIDEO_BUCKET, "video/mp4")
            except Exception as e:
                errors.append(f"video: {e}")
        else:
            errors.append(f"video: {video_path} not found")
        if image_path.exists():
            try:
                updates["images"] = [storage.upload_file(str(image_path), slug, IMAGE_BUCKET, "image/png")]
            except Exception as e:
                errors.append(f"image: {e}")
        else:
            errors.append(f"image: {image_path} not found")

        if updates:
            assignments = ", ".join(f"{column} = %s" for column in updates)
            with conn.cursor() as cur:
                cur.execute(f"UPDATE exercises SET {assignments}, updated_at = CURRENT_TIMESTAMP WHERE slug = %s",
                            (*updates.values(), slug))
            conn.commit()
    return {"updated": updates, "errors": errors}

def main() -> None:
    load_dotenv()
    slugs = sys.argv[1:]
    if not slugs:
        raise SystemExit("Usage: python upload_media.py <slug> [<slug> ...]")
    storage = SupabaseStorageClient()
    for slug in slugs:
        try:
            result = repair(slug, storage)
        except Exception as e:
            print(f"❌ {slug}: {e}")
            continue
        for column, value in result["updated"].items():
            print(f"✅ {slug}: {column} -> {value}")
        for error in result["errors"]:
            print(f"⚠️ {slug}: {error}")

if __name__ == "__main__":
    main()
