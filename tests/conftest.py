from src.domain.models import ExerciseModel


def make_exercise(**overrides) -> ExerciseModel:
    """Builds a valid ExerciseModel; keyword arguments override the defaults."""
    data = {
        "name": "Romanian Deadlift",
        "slug": "llm_made_up_slug",
        "force": "pull",
        "mechanic": "compound",
        "equipment": "barbell",
        "category": "strength",
        "primary_muscles": ["hamstrings"],
        "secondary_muscles": ["glutes"],
        "preparation": "Stand tall holding the bar.",
        "execution_steps": ["Hinge at the hips.", "Return to standing."],
        "breathing": "Inhale down, exhale up.",
        "common_mistakes": ["Rounding the back."],
    }
    data.update(overrides)
    return ExerciseModel(**data)
