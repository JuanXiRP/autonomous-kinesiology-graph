from datetime import datetime

from src.cli.prompts import choose_variants, parse_selection, print_options
from src.domain.models import EquipmentType
from src.resolution.variants import ExerciseVariant, VariantOptions


def options() -> VariantOptions:
    return VariantOptions(interpreted_as_es="Peso muerto rumano", variants=[
        ExerciseVariant(name="Romanian Deadlift", equipment=EquipmentType.BARBELL,
                        label_es="Peso muerto rumano (barra)", note_es="Clásico."),
        ExerciseVariant(name="Romanian Deadlift", equipment=EquipmentType.DUMBBELL,
                        label_es="Peso muerto rumano (mancuernas)", note_es="Más recorrido."),
        ExerciseVariant(name="Single-Leg Romanian Deadlift", equipment=EquipmentType.DUMBBELL,
                        label_es="Peso muerto rumano unilateral (mancuerna)", note_es="Equilibrio."),
    ])


def scripted(*answers):
    queue = list(answers)
    return lambda prompt: queue.pop(0)


def test_parse_selection_accepts_lists_ranges_and_all():
    # Arrange / Act / Assert
    assert parse_selection("1,3", 3) == [0, 2]
    assert parse_selection("1-3", 3) == [0, 1, 2]
    assert parse_selection("2 3", 3) == [1, 2]
    assert parse_selection("todos", 3) == [0, 1, 2]
    assert parse_selection("1,1", 3) == [0]


def test_parse_selection_rejects_invalid_input():
    # Arrange / Act / Assert
    assert parse_selection("4", 3) is None
    assert parse_selection("a", 3) is None
    assert parse_selection("3-1", 3) is None


def test_empty_selection_cancels():
    # Arrange / Act / Assert
    assert parse_selection("", 3) == []


def test_invalid_answer_is_asked_again():
    # Arrange
    printed = []

    # Act
    selected = choose_variants(options(), {}, True, scripted("9", "2"), printed.append)

    # Assert
    assert [v.slug for v in selected] == ["romanian_deadlift_dumbbell"]
    assert any("no válida" in line for line in printed)


def test_existing_variant_needs_confirmation_to_overwrite():
    # Arrange
    existing = {"romanian_deadlift_barbell": datetime(2026, 10, 8)}

    # Act
    declined = choose_variants(options(), existing, True, scripted("1,2", "n"), lambda *_: None)
    accepted = choose_variants(options(), existing, True, scripted("1", "s"), lambda *_: None)

    # Assert
    assert [v.slug for v in declined] == ["romanian_deadlift_dumbbell"]
    assert [v.slug for v in accepted] == ["romanian_deadlift_barbell"]


def test_unchecked_database_asks_before_continuing():
    # Arrange / Act
    selected = choose_variants(options(), {}, False, scripted("1", "n"), lambda *_: None)

    # Assert
    assert selected == []


def test_options_mark_variants_already_in_the_database():
    # Arrange
    printed = []
    existing = {"romanian_deadlift_barbell": datetime(2026, 10, 8)}

    # Act
    print_options(options(), existing, ["Good Morning (Barbell)"], printed.append)

    # Assert
    text = "\n".join(printed)
    assert "[ya en BD, 08/10/2026]" in text
    assert text.count("ya en BD") == 1
    assert "Good Morning (Barbell)" in text
