from langchain_core.runnables import RunnableLambda

from src.domain.animation import MovementFamily
from src.domain.models import EquipmentType
from src.resolution import variants
from src.resolution.variants import ExerciseVariant, VariantOptions


def make_variant(name="Romanian Deadlift", equipment=EquipmentType.BARBELL, label="Peso muerto rumano (barra)"):
    return ExerciseVariant(name=name, equipment=equipment, label_es=label, note_es="La versión clásica.")


class FakeLLM:
    def __init__(self, result):
        self.result = result

    def with_structured_output(self, schema):
        return RunnableLambda(lambda _: self.result)


def test_variant_slug_matches_the_database_rule():
    # Arrange / Act
    variant = make_variant()

    # Assert
    assert variant.slug == "romanian_deadlift_barbell"
    assert variant.full_name == "Romanian Deadlift (Barbell)"


def test_resolver_removes_duplicate_slugs(monkeypatch):
    # Arrange
    options = VariantOptions(interpreted_as_es="Peso muerto rumano", variants=[
        make_variant(),
        make_variant(label="Peso muerto rumano con barra"),
        make_variant(equipment=EquipmentType.DUMBBELL, label="Peso muerto rumano (mancuernas)"),
    ])
    monkeypatch.setattr(variants, "ChatGoogleGenerativeAI", lambda **kwargs: FakeLLM(options))

    # Act
    result = variants.resolve_variants("peso muerto rumano")

    # Assert
    assert [v.slug for v in result.variants] == ["romanian_deadlift_barbell", "romanian_deadlift_dumbbell"]


def test_movement_family_defaults_to_other():
    # Arrange / Act
    variant = make_variant()

    # Assert
    assert variant.movement_family == MovementFamily.OTHER


def test_movement_family_is_parsed():
    # Arrange / Act
    variant = ExerciseVariant(name="Bench Press", equipment=EquipmentType.BARBELL, label_es="Press banca (barra)",
                              note_es="Clásico.", movement_family="horizontal_press")

    # Assert
    assert variant.movement_family == MovementFamily.HORIZONTAL_PRESS
