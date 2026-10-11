from src.graph import app, route_after_generation, route_after_spec
from tests.conftest import make_exercise


def test_no_exercise_with_errors_and_attempts_left_retries_content():
    # Arrange
    state = {"structured_exercise": None, "validation_errors": ["boom"], "generation_attempts": 1}

    # Act / Assert
    assert route_after_generation(state) == "content_generator"


def test_no_exercise_after_max_attempts_ends():
    # Arrange
    state = {"structured_exercise": None, "validation_errors": ["boom"], "generation_attempts": 3}

    # Act / Assert
    assert route_after_generation(state) == "__end__"


def test_exercise_goes_to_spec_generation():
    # Arrange
    state = {"structured_exercise": make_exercise(), "validation_errors": []}

    # Act / Assert
    assert route_after_generation(state) == "spec_generator"


def test_spec_present_goes_to_render():
    # Arrange
    state = {"animation_spec": object(), "spec_attempts": 1}

    # Act / Assert
    assert route_after_spec(state) == "render_generator"


def test_missing_spec_retries_until_attempts_run_out():
    # Arrange / Act / Assert
    assert route_after_spec({"animation_spec": None, "spec_attempts": 2}) == "spec_generator"
    assert route_after_spec({"animation_spec": None, "spec_attempts": 3}) == "storage_uploader"


def test_render_goes_through_storage_before_database():
    # Arrange / Act
    edges = {(e.source, e.target) for e in app.get_graph().edges}

    # Assert
    assert ("render_generator", "storage_uploader") in edges
    assert ("storage_uploader", "db_writer") in edges
    assert ("render_generator", "db_writer") not in edges
    assert not any("video_generator" in edge or "media_generator" in edge for edge in edges)
