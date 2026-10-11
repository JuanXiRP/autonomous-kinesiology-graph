from datetime import datetime
from types import SimpleNamespace

import pytest

from src.infrastructure.db import exercise_repository as repo


class FakeCursor:
    def __init__(self, captured, rows):
        self.captured, self.rows = captured, rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params):
        self.captured["query"], self.captured["params"] = query, params

    def fetchall(self):
        return self.rows


class FakeConn:
    def __init__(self, captured, rows):
        self.captured, self.rows = captured, rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return FakeCursor(self.captured, self.rows)


def install(monkeypatch, rows, captured):
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake")
    monkeypatch.setattr(repo, "psycopg", SimpleNamespace(connect=lambda url: FakeConn(captured, rows)))


def test_existing_slugs_uses_a_single_any_query(monkeypatch):
    # Arrange
    captured, when = {}, datetime(2026, 10, 8)
    install(monkeypatch, [("romanian_deadlift_barbell", when)], captured)

    # Act
    result = repo.existing_slugs(["romanian_deadlift_barbell", "romanian_deadlift_dumbbell"])

    # Assert
    assert result == {"romanian_deadlift_barbell": when}
    assert "ANY(%s)" in captured["query"]
    assert captured["params"] == (["romanian_deadlift_barbell", "romanian_deadlift_dumbbell"],)


def test_existing_slugs_raises_when_the_database_is_unreachable(monkeypatch):
    # Arrange
    def boom(url):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(repo, "psycopg", SimpleNamespace(connect=boom))

    # Act / Assert
    with pytest.raises(RuntimeError):
        repo.existing_slugs(["x"])


def test_similar_names_returns_empty_list_on_error(monkeypatch):
    # Arrange
    def boom(url):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(repo, "psycopg", SimpleNamespace(connect=boom))

    # Act / Assert
    assert repo.similar_names("Romanian Deadlift") == []


def test_similar_names_uses_ilike(monkeypatch):
    # Arrange
    captured = {}
    install(monkeypatch, [("Romanian Deadlift (Barbell)",)], captured)

    # Act
    names = repo.similar_names("Romanian Deadlift")

    # Assert
    assert names == ["Romanian Deadlift (Barbell)"]
    assert "ILIKE" in captured["query"]
    assert captured["params"][0] == "%Romanian Deadlift%"
