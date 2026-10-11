import main


def set_all_env(monkeypatch):
    for name in main.REQUIRED_ENV:
        monkeypatch.setenv(name, "dummy")


def test_missing_env_lists_unset_name(monkeypatch):
    # Arrange
    set_all_env(monkeypatch)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)

    # Act
    missing = main.missing_env()

    # Assert
    assert missing == ["SUPABASE_KEY"]


def test_empty_value_counts_as_missing(monkeypatch):
    # Arrange
    set_all_env(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "")

    # Act / Assert
    assert main.missing_env() == ["DATABASE_URL"]


def test_interactive_stops_when_env_is_missing(monkeypatch, capsys):
    # Arrange
    def must_not_run(query):
        raise AssertionError("resolve_variants must not be called")

    set_all_env(monkeypatch)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(main, "resolve_variants", must_not_run)

    # Act
    main.interactive("peso muerto rumano")

    # Assert
    out = capsys.readouterr().out
    assert "GOOGLE_API_KEY" in out
    assert "dummy" not in out


def test_interactive_generates_only_the_chosen_variants(monkeypatch):
    # Arrange
    from datetime import datetime
    from tests.test_cli_selection import options

    set_all_env(monkeypatch)
    ran = {}
    monkeypatch.setattr(main, "resolve_variants", lambda query: options())
    monkeypatch.setattr(main, "existing_slugs", lambda slugs: {"romanian_deadlift_barbell": datetime(2026, 10, 8)})
    monkeypatch.setattr(main, "similar_names", lambda term: [])
    monkeypatch.setattr(main, "run_variants", lambda selected, existing: ran.update(
        selected=[v.slug for v in selected], existing=existing) or [])
    answers = iter(["1,3", "n", "s"])  # pick 1 and 3, refuse to overwrite 1, confirm generation

    # Act
    main.interactive("peso muerto rumano", input_fn=lambda prompt: next(answers))

    # Assert
    assert ran["selected"] == ["single-leg_romanian_deadlift_dumbbell"]
    assert ran["existing"] == {"romanian_deadlift_barbell"}


def test_interactive_does_nothing_when_cancelled(monkeypatch):
    # Arrange
    from tests.test_cli_selection import options

    set_all_env(monkeypatch)
    monkeypatch.setattr(main, "resolve_variants", lambda query: options())
    monkeypatch.setattr(main, "existing_slugs", lambda slugs: {})
    monkeypatch.setattr(main, "similar_names", lambda term: [])

    def must_not_run(*args, **kwargs):
        raise AssertionError("run_variants must not be called")

    monkeypatch.setattr(main, "run_variants", must_not_run)

    # Act / Assert (empty selection cancels)
    main.interactive("peso muerto rumano", input_fn=lambda prompt: "")


def test_interactive_survives_an_unreachable_database(monkeypatch):
    # Arrange
    from tests.test_cli_selection import options

    set_all_env(monkeypatch)
    monkeypatch.setattr(main, "resolve_variants", lambda query: options())

    def boom(slugs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(main, "existing_slugs", boom)
    monkeypatch.setattr(main, "similar_names", lambda term: [])
    ran = {}
    monkeypatch.setattr(main, "run_variants", lambda selected, existing: ran.update(n=len(selected)) or [])
    answers = iter(["2", "s", "s"])  # pick 2, accept the unchecked-database warning, confirm

    # Act
    main.interactive("peso muerto rumano", input_fn=lambda prompt: next(answers))

    # Assert
    assert ran["n"] == 1
