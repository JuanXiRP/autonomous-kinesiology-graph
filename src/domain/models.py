# src/domain/models.py
from enum import StrEnum
from pydantic import BaseModel, Field, model_validator

class EquipmentType(StrEnum):
    MACHINE = "machine"
    DUMBBELL = "dumbbell"
    BARBELL = "barbell"
    CABLE = "cable"
    BODYWEIGHT = "bodyweight"
    KETTLEBELL = "kettlebell"
    BAND = "band"

class ForceType(StrEnum):
    PULL = "pull"
    PUSH = "push"
    STATIC = "static"

class MechanicType(StrEnum):
    ISOLATION = "isolation"
    COMPOUND = "compound"

class CategoryType(StrEnum):
    STRENGTH = "strength"
    CARDIO = "cardio"
    STRETCHING = "stretching"
    PLYOMETRICS = "plyometrics"

def display_name(name: str, equipment: "EquipmentType") -> str:
    """Base name plus the equipment in brackets, e.g. 'Romanian Deadlift (Barbell)'."""
    formatted_equipment = f"({equipment.value.capitalize()})"
    if name.endswith(formatted_equipment):
        return name
    return f"{name.strip()} {formatted_equipment}"

def make_slug(name: str, equipment: "EquipmentType") -> str:
    """Upsert key and Storage folder. Changing this rule orphans existing rows and files."""
    return display_name(name, equipment).replace('(', '').replace(')', '').replace(' ', '_').lower()

class ExerciseModel(BaseModel):
    name: str = Field(description="The standard base name of the exercise WITHOUT the equipment type.")
    slug: str = Field(description="URL-friendly version of the name, lowercase with underscores.")
    
    force: ForceType = Field(description="The primary type of force applied.")
    mechanic: MechanicType = Field(description="Whether the movement isolates a joint or uses multiple.")
    equipment: EquipmentType = Field(description="The primary equipment required.")
    category: CategoryType = Field(description="The broad category of the exercise.")
    
    primary_muscles: list[str] = Field(description="List of primary activated muscles.")
    secondary_muscles: list[str] = Field(description="List of synergist or stabilizer muscles.")
    
    preparation: str = Field(description="Starting position and setup instructions.")
    execution_steps: list[str] = Field(description="Step-by-step movement execution.")
    breathing: str = Field(description="Specific breathing instructions.")
    common_mistakes: list[str] = Field(description="2-3 common mistakes to avoid.")
    
    images: list[str] = Field(default_factory=list, description="Paths to generated keyframe images.")
    animation_video: str | None = Field(default=None, description="Path to the generated stick-figure animation video.")

    @model_validator(mode='after')
    def append_equipment_to_name(self) -> 'ExerciseModel':
        # Append equipment to name to ensure uniqueness and consistency
        self.name = display_name(self.name, self.equipment)

        # Ensure slug matches the final sanitized name
        self.slug = make_slug(self.name, self.equipment)
        
        return self