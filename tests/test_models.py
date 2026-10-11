from tests.conftest import make_exercise


def test_validator_appends_equipment_to_name():
    # Arrange / Act
    exercise = make_exercise(name="Romanian Deadlift", equipment="barbell")

    # Assert
    assert exercise.name == "Romanian Deadlift (Barbell)"


def test_validator_does_not_append_equipment_twice():
    # Arrange / Act
    exercise = make_exercise(name="Romanian Deadlift (Barbell)", equipment="barbell")

    # Assert
    assert exercise.name == "Romanian Deadlift (Barbell)"


def test_slug_is_rebuilt_from_final_name_ignoring_llm_slug():
    # Arrange / Act
    exercise = make_exercise(slug="something_else")

    # Assert
    assert exercise.slug == "romanian_deadlift_barbell"


def test_make_slug_is_pinned_for_existing_rows():
    # Arrange
    from src.domain.models import EquipmentType, make_slug

    # Act / Assert
    assert make_slug("Romanian Deadlift", EquipmentType.BARBELL) == "romanian_deadlift_barbell"
    assert make_slug("Romanian Deadlift (Barbell)", EquipmentType.BARBELL) == "romanian_deadlift_barbell"
