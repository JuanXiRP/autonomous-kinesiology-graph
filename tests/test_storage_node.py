from src.nodes import storage_node
from tests.conftest import make_exercise


class FakeStorage:
    calls = []
    fail_buckets = ()

    def upload_file(self, local_path, slug, bucket, content_type):
        FakeStorage.calls.append((bucket, content_type))
        if bucket in FakeStorage.fail_buckets:
            raise RuntimeError("upload failed")
        return f"https://example.test/{bucket}/{slug}/{local_path}"


def install(monkeypatch, fail_buckets=()):
    FakeStorage.calls = []
    FakeStorage.fail_buckets = fail_buckets
    monkeypatch.setattr(storage_node, "SupabaseStorageClient", FakeStorage)


def test_success_replaces_paths_with_urls(monkeypatch):
    # Arrange
    install(monkeypatch)
    exercise = make_exercise(images=["a.jpg"], animation_video="v.mp4")

    # Act
    result = storage_node.upload_exercise_media({"structured_exercise": exercise})

    # Assert
    out = result["structured_exercise"]
    assert out.animation_video.startswith("https://")
    assert out.images[0].startswith("https://")
    assert result["validation_errors"] == []
    assert ("exercise_videos", "video/mp4") in FakeStorage.calls
    assert ("exercise_images", "image/jpeg") in FakeStorage.calls


def test_video_failure_clears_video_but_uploads_images(monkeypatch):
    # Arrange
    install(monkeypatch, fail_buckets=("exercise_videos",))
    exercise = make_exercise(images=["a.jpg"], animation_video="v.mp4")

    # Act
    result = storage_node.upload_exercise_media({"structured_exercise": exercise})

    # Assert
    out = result["structured_exercise"]
    assert out.animation_video is None
    assert out.images[0].startswith("https://")
    assert len(result["validation_errors"]) == 1


def test_client_construction_failure_clears_local_paths(monkeypatch):
    # Arrange
    def boom():
        raise RuntimeError("no credentials")

    monkeypatch.setattr(storage_node, "SupabaseStorageClient", boom)
    exercise = make_exercise(images=["a.jpg"], animation_video="v.mp4")

    # Act
    result = storage_node.upload_exercise_media({"structured_exercise": exercise})

    # Assert
    out = result["structured_exercise"]
    assert out.images == []
    assert out.animation_video is None
    assert len(result["validation_errors"]) == 1


def test_missing_exercise_reports_error():
    # Arrange / Act
    result = storage_node.upload_exercise_media({"structured_exercise": None})

    # Assert
    assert result == {"validation_errors": ["StorageNode failed: No exercise data."]}
