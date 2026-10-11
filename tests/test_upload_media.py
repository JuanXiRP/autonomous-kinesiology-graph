import pytest

import upload_media


class FakeStorage:
    def __init__(self, fail_buckets=()):
        self.calls, self.fail_buckets = [], fail_buckets

    def upload_file(self, local_path, slug, bucket, content_type):
        self.calls.append((bucket, content_type))
        if bucket in self.fail_buckets:
            raise RuntimeError("upload failed")
        return f"https://example.test/{bucket}/{slug}"


class FakeCursor:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params):
        self.db["queries"].append((query, params))

    def fetchone(self):
        return (1,) if self.db["row_exists"] else None


class FakeConn:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return FakeCursor(self.db)

    def commit(self):
        self.db["committed"] = True


def setup_files(tmp_path, monkeypatch, video=True, image=True):
    monkeypatch.chdir(tmp_path)
    video_path, image_path = upload_media.local_media("slug")
    for path, wanted in ((video_path, video), (image_path, image)):
        if wanted:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"data")


def make_db(row_exists=True):
    return {"queries": [], "row_exists": row_exists, "committed": False}


def test_uploads_both_files_and_updates_the_row(tmp_path, monkeypatch):
    # Arrange
    setup_files(tmp_path, monkeypatch)
    db, storage = make_db(), FakeStorage()

    # Act
    result = upload_media.repair("slug", storage, connect=lambda url: FakeConn(db))

    # Assert
    update_query, params = db["queries"][-1]
    assert "animation_video = %s" in update_query and "images = %s" in update_query
    assert params[-1] == "slug"
    assert db["committed"]
    assert result["errors"] == []


def test_only_uploaded_fields_are_updated(tmp_path, monkeypatch):
    # Arrange
    setup_files(tmp_path, monkeypatch)
    db, storage = make_db(), FakeStorage(fail_buckets={"exercise_videos"})

    # Act
    result = upload_media.repair("slug", storage, connect=lambda url: FakeConn(db))

    # Assert
    update_query, _ = db["queries"][-1]
    assert "animation_video" not in update_query and "images = %s" in update_query
    assert result["errors"] == ["video: upload failed"]


def test_unknown_slug_uploads_nothing(tmp_path, monkeypatch):
    # Arrange
    setup_files(tmp_path, monkeypatch)
    db, storage = make_db(row_exists=False), FakeStorage()

    # Act / Assert
    with pytest.raises(LookupError):
        upload_media.repair("slug", storage, connect=lambda url: FakeConn(db))
    assert storage.calls == []


def test_missing_local_files_are_reported_without_an_update(tmp_path, monkeypatch):
    # Arrange
    setup_files(tmp_path, monkeypatch, video=False, image=False)
    db = make_db()

    # Act
    result = upload_media.repair("slug", FakeStorage(), connect=lambda url: FakeConn(db))

    # Assert
    assert len(result["errors"]) == 2
    assert not any(query.startswith("UPDATE") for query, _ in db["queries"])
