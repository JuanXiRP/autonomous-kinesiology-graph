from types import SimpleNamespace

import pytest

from src.infrastructure.db import storage_client
from src.infrastructure.db.storage_client import SupabaseStorageClient


class FakeBucket:
    def __init__(self, uploads):
        self.uploads = uploads

    def upload(self, file, path, file_options):
        self.uploads.append((path, file_options))

    def get_public_url(self, path):
        return f"https://example.test/{path}"


def fake_head(status=200, size=4):
    def head(url, **kwargs):
        return SimpleNamespace(status_code=status, headers={"content-length": str(size)})
    return head


@pytest.fixture(autouse=True)
def remote_matches_local(monkeypatch):
    monkeypatch.setattr(storage_client.httpx, "head", fake_head())


def make_client():
    uploads = []
    client = object.__new__(SupabaseStorageClient)
    client.supabase = SimpleNamespace(
        storage=SimpleNamespace(from_=lambda bucket: FakeBucket(uploads))
    )
    return client, uploads


def test_empty_file_is_rejected_without_uploading(tmp_path):
    # Arrange
    client, uploads = make_client()
    empty = tmp_path / "video.mp4"
    empty.write_bytes(b"")

    # Act / Assert
    with pytest.raises(ValueError):
        client.upload_file(str(empty), "slug", "bucket", "video/mp4")
    assert uploads == []


def test_missing_file_raises_file_not_found(tmp_path):
    # Arrange
    client, uploads = make_client()

    # Act / Assert
    with pytest.raises(FileNotFoundError):
        client.upload_file(str(tmp_path / "nope.mp4"), "slug", "bucket", "video/mp4")
    assert uploads == []


def test_successful_upload_returns_public_url(tmp_path):
    # Arrange
    client, uploads = make_client()
    video = tmp_path / "video.mp4"
    video.write_bytes(b"data")

    # Act
    url = client.upload_file(str(video), "slug", "bucket", "video/mp4")

    # Assert
    assert url == "https://example.test/slug/video.mp4"
    assert uploads == [("slug/video.mp4", {"content-type": "video/mp4", "upsert": "true"})]


def test_stale_remote_object_is_reported(tmp_path, monkeypatch):
    # Arrange
    client, _ = make_client()
    video = tmp_path / "video.mp4"
    video.write_bytes(b"data")
    monkeypatch.setattr(storage_client.httpx, "head", fake_head(size=999))

    # Act / Assert
    with pytest.raises(RuntimeError, match="size"):
        client.upload_file(str(video), "slug", "bucket", "video/mp4")


def test_unreadable_remote_object_is_reported(tmp_path, monkeypatch):
    # Arrange
    client, _ = make_client()
    video = tmp_path / "video.mp4"
    video.write_bytes(b"data")
    monkeypatch.setattr(storage_client.httpx, "head", fake_head(status=404))

    # Act / Assert
    with pytest.raises(RuntimeError, match="404"):
        client.upload_file(str(video), "slug", "bucket", "video/mp4")


class FlakyBucket(FakeBucket):
    def __init__(self, uploads, failures):
        super().__init__(uploads)
        self.failures = failures

    def upload(self, file, path, file_options):
        if self.failures:
            raise self.failures.pop(0)
        super().upload(file, path, file_options)


def make_flaky_client(failures):
    uploads = []
    bucket = FlakyBucket(uploads, failures)
    client = object.__new__(SupabaseStorageClient)
    client.supabase = SimpleNamespace(storage=SimpleNamespace(from_=lambda name: bucket))
    return client, uploads


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(storage_client.time, "sleep", lambda s: None)


def test_transport_errors_are_retried(tmp_path, no_sleep):
    # Arrange
    import httpx
    client, uploads = make_flaky_client([httpx.RemoteProtocolError("Server disconnected"),
                                         httpx.ConnectError("reset")])
    video = tmp_path / "video.mp4"
    video.write_bytes(b"data")

    # Act
    url = client.upload_file(str(video), "slug", "bucket", "video/mp4")

    # Assert
    assert url.endswith("slug/video.mp4")
    assert len(uploads) == 1


def test_ssl_error_is_retried(tmp_path, no_sleep):
    # Arrange
    import ssl
    client, uploads = make_flaky_client([ssl.SSLError("bad record mac")])
    video = tmp_path / "video.mp4"
    video.write_bytes(b"data")

    # Act
    client.upload_file(str(video), "slug", "bucket", "video/mp4")

    # Assert
    assert len(uploads) == 1


def test_gives_up_after_three_transport_errors(tmp_path, no_sleep):
    # Arrange
    import httpx
    client, uploads = make_flaky_client([httpx.ReadTimeout("t") for _ in range(3)])
    video = tmp_path / "video.mp4"
    video.write_bytes(b"data")

    # Act / Assert
    with pytest.raises(httpx.ReadTimeout):
        client.upload_file(str(video), "slug", "bucket", "video/mp4")
    assert uploads == []


def test_non_transport_errors_are_not_retried(tmp_path, no_sleep):
    # Arrange
    client, uploads = make_flaky_client([RuntimeError("403 forbidden"), None])
    video = tmp_path / "video.mp4"
    video.write_bytes(b"data")

    # Act / Assert
    with pytest.raises(RuntimeError, match="403"):
        client.upload_file(str(video), "slug", "bucket", "video/mp4")
    assert uploads == []


def test_uploads_from_several_threads_never_overlap(tmp_path):
    # Arrange
    import threading
    import time as real_time
    active, peak, lock = [0], [0], threading.Lock()

    class SlowBucket(FakeBucket):
        def upload(self, file, path, file_options):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            real_time.sleep(0.03)
            with lock:
                active[0] -= 1

    client = object.__new__(SupabaseStorageClient)
    client.supabase = SimpleNamespace(storage=SimpleNamespace(from_=lambda name: SlowBucket([])))
    files = []
    for i in range(4):
        path = tmp_path / f"v{i}.mp4"
        path.write_bytes(b"data")
        files.append(path)

    # Act
    threads = [threading.Thread(target=client.upload_file, args=(str(p), "slug", "bucket", "video/mp4")) for p in files]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Assert
    assert peak[0] == 1
