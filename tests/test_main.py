import main


def set_all_env(monkeypatch):
    for name in main.REQUIRED_ENV + main.MEDIA_ENV:
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
    monkeypatch.setattr(main, "run_variants", lambda selected, existing, with_media: ran.update(
        selected=[v.slug for v in selected], existing=existing, with_media=with_media) or [])
    answers = iter(["1,3", "n", "n", "s"])  # pick 1 and 3, refuse to overwrite 1, text only, confirm

    # Act
    main.interactive("peso muerto rumano", input_fn=lambda prompt: next(answers))

    # Assert
    assert ran["selected"] == ["single-leg_romanian_deadlift_dumbbell"]
    assert ran["existing"] == {"romanian_deadlift_barbell"}
    assert ran["with_media"] is False


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
    monkeypatch.setattr(main, "run_variants", lambda selected, existing, with_media: ran.update(n=len(selected)) or [])
    answers = iter(["2", "s", "n", "s"])  # pick 2, accept the unchecked-database warning, text only, confirm

    # Act
    main.interactive("peso muerto rumano", input_fn=lambda prompt: next(answers))

    # Assert
    assert ran["n"] == 1


def test_parse_args_extracts_the_media_flags():
    # Arrange / Act / Assert
    assert main.parse_args([]) == (None, None)
    assert main.parse_args(["peso", "muerto"]) == ("peso muerto", None)
    assert main.parse_args(["--solo-texto", "peso muerto rumano"]) == ("peso muerto rumano", False)
    assert main.parse_args(["peso", "--con-video"]) == ("peso", True)


def test_text_only_does_not_need_storage_credentials(monkeypatch):
    # Arrange
    from tests.test_cli_selection import options

    set_all_env(monkeypatch)
    for name in main.MEDIA_ENV:
        monkeypatch.delenv(name, raising=False)
    ran = {}
    monkeypatch.setattr(main, "resolve_variants", lambda query: options())
    monkeypatch.setattr(main, "existing_slugs", lambda slugs: {})
    monkeypatch.setattr(main, "similar_names", lambda term: [])
    monkeypatch.setattr(main, "run_variants", lambda selected, existing, with_media: ran.update(
        with_media=with_media) or [])
    answers = iter(["2", "s"])  # pick 2, confirm (the flag skips the media question)

    # Act
    main.interactive("peso muerto rumano", input_fn=lambda prompt: next(answers), with_media=False)

    # Assert
    assert ran["with_media"] is False


def test_media_mode_still_requires_storage_credentials(monkeypatch, capsys):
    # Arrange
    from tests.test_cli_selection import options

    set_all_env(monkeypatch)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.setattr(main, "resolve_variants", lambda query: options())
    monkeypatch.setattr(main, "existing_slugs", lambda slugs: {})
    monkeypatch.setattr(main, "similar_names", lambda term: [])

    def must_not_run(*args, **kwargs):
        raise AssertionError("run_variants must not be called")

    monkeypatch.setattr(main, "run_variants", must_not_run)
    answers = iter(["2", "s"])  # pick 2, ask for video and still

    # Act
    main.interactive("peso muerto rumano", input_fn=lambda prompt: next(answers))

    # Assert
    assert "SUPABASE_KEY" in capsys.readouterr().out
