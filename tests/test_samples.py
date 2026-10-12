import json
from pathlib import Path

import pytest

from src.domain.animation import AnimationSpec, MovementFamily
from src.rendering import checks

SAMPLES = Path("samples/specs")
INDEX = json.loads((SAMPLES / "index.json").read_text(encoding="utf-8"))


def test_every_sample_file_is_indexed():
    # Arrange / Act
    files = {path.name for path in SAMPLES.glob("*.json")} - {"index.json"}

    # Assert
    assert files == {entry["file"] for entry in INDEX}


@pytest.mark.parametrize("entry", INDEX, ids=lambda e: e["file"])
def test_sample_is_valid_and_has_no_geometry_problems(entry):
    # Arrange
    spec = AnimationSpec.model_validate(json.loads((SAMPLES / entry["file"]).read_text(encoding="utf-8")))

    # Act
    problems = checks.check_spec_geometry(spec, entry["equipment"])

    # Assert
    assert problems == []


def test_sample_families_are_known_movement_families():
    # Arrange / Act
    families = {entry["family"] for entry in INDEX}

    # Assert
    assert families <= {family.value for family in MovementFamily}
