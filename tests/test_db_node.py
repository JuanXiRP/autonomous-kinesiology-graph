from types import SimpleNamespace

from src.nodes import db_node
from tests.conftest import make_exercise


class FakeCursor:
    def __init__(self, captured):
        self.captured = captured

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, values):
        self.captured["query"] = query
        self.captured["values"] = values

    def fetchone(self):
        return ("fake-id",)


class FakeConn:
    def __init__(self, captured):
        self.captured = captured

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return FakeCursor(self.captured)

    def commit(self):
        pass


def install_fake_connect(monkeypatch, captured):
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake")
    monkeypatch.setattr(
        db_node, "psycopg", SimpleNamespace(connect=lambda url: FakeConn(captured))
    )


def test_upsert_query_refreshes_every_column(monkeypatch):
    # Arrange
    captured = {}
    install_fake_connect(monkeypatch, captured)
    exercise = make_exercise(images=["https://x/a.jpg"], animation_video="https://x/v.mp4")

    # Act
    result = db_node.save_to_database({"structured_exercise": exercise})

    # Assert
    query = captured["query"]
    assert "force = EXCLUDED.force" in query
    assert "COALESCE(EXCLUDED.animation_video" in query
    assert "cardinality(EXCLUDED.images)" in query
    assert "updated_at = CURRENT_TIMESTAMP" in query
    assert result == {}


def test_values_follow_insert_column_order(monkeypatch):
    # Arrange
    captured = {}
    install_fake_connect(monkeypatch, captured)
    exercise = make_exercise(images=["https://x/a.jpg"], animation_video="https://x/v.mp4")

    # Act
    db_node.save_to_database({"structured_exercise": exercise})

    # Assert
    assert captured["values"] == (
        exercise.slug, exercise.name, "pull", "compound", "barbell", "strength",
        exercise.primary_muscles, exercise.secondary_muscles,
        exercise.preparation, exercise.execution_steps, exercise.breathing,
        exercise.common_mistakes, exercise.images, exercise.animation_video,
    )


def test_connection_failure_is_reported_not_raised(monkeypatch):
    # Arrange
    def boom(url):
        raise RuntimeError("connection refused")

    monkeypatch.setenv("DATABASE_URL", "postgresql://fake")
    monkeypatch.setattr(db_node, "psycopg", SimpleNamespace(connect=boom))

    # Act
    result = db_node.save_to_database({"structured_exercise": make_exercise()})

    # Assert
    assert len(result["validation_errors"]) == 1
    assert "connection refused" in result["validation_errors"][0]
