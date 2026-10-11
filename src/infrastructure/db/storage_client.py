import os
import ssl
import logging
import threading
import time
import httpx
from supabase import create_client, Client, ClientOptions
from pathlib import Path

VIDEO_BUCKET = "exercise_videos"
IMAGE_BUCKET = "exercise_images"

# Uploads failed with "Server disconnected" / SSL "bad record mac" when two variants uploaded videos at
# the same time over storage3's default HTTP/2 client. Use HTTP/1.1, retry transport errors and
# serialise uploads across threads (they take seconds; the renders stay parallel).
UPLOAD_ATTEMPTS = 3
RETRY_DELAYS_S = (2, 5)
UPLOAD_TIMEOUT_S = 120
UPLOAD_LOCK = threading.Lock()
TRANSIENT_ERRORS = (httpx.TransportError, ssl.SSLError, ConnectionError)

class SupabaseStorageClient:
    def __init__(self):
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")
        http_client = httpx.Client(http2=False, timeout=UPLOAD_TIMEOUT_S, follow_redirects=True)
        self.supabase: Client = create_client(url, key, options=ClientOptions(httpx_client=http_client))

    def upload_file(self, local_path: str, slug: str, bucket: str, content_type: str) -> str:
        """Uploads a local file to Supabase Storage and returns its public URL."""
        file_path = Path(local_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")
        if file_path.stat().st_size == 0:
            raise ValueError(f"Refusing to upload an empty file: {local_path}")

        storage_path = f"{slug}/{file_path.name}"

        with UPLOAD_LOCK:
            for attempt in range(1, UPLOAD_ATTEMPTS + 1):
                try:
                    logging.info(f"☁️ [Storage] Uploading {file_path.name} to bucket '{bucket}' (attempt {attempt})...")
                    # Upload the file (overwrites if it already exists)
                    with open(file_path, "rb") as f:
                        self.supabase.storage.from_(bucket).upload(
                            file=f,
                            path=storage_path,
                            file_options={"content-type": content_type, "upsert": "true"}
                        )
                    break
                except TRANSIENT_ERRORS as e:
                    if attempt == UPLOAD_ATTEMPTS:
                        raise
                    delay = RETRY_DELAYS_S[min(attempt - 1, len(RETRY_DELAYS_S) - 1)]
                    logging.warning(f"⚠️ [Storage] {slug}: upload attempt {attempt} failed ({e}); retrying in {delay} s")
                    time.sleep(delay)

        public_url = self.supabase.storage.from_(bucket).get_public_url(storage_path)
        self._verify_remote_size(public_url, file_path.stat().st_size)
        logging.info(f"✅ [Storage] Upload completed. URL: {public_url}")

        return public_url

    @staticmethod
    def _verify_remote_size(public_url: str, expected_size: int) -> None:
        """Fails loudly when the public object is not the file just uploaded (stale or missing)."""
        response = httpx.head(public_url, timeout=20, follow_redirects=True)
        if response.status_code != 200:
            raise RuntimeError(f"Uploaded object is not publicly readable (HTTP {response.status_code}): {public_url}")
        remote_size = int(response.headers.get("content-length", -1))
        if remote_size != expected_size:
            raise RuntimeError(
                f"Remote object size {remote_size} != local file size {expected_size}: {public_url}"
            )
