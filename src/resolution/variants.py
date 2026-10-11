# src/resolution/variants.py
# Turns a free-text request ("peso muerto rumano") into the concrete exercise variants the user can pick.
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.domain.models import EquipmentType, display_name, make_slug

# This model is the Gemini structured-output schema: the Field descriptions are prompt text.

class ExerciseVariant(BaseModel):
    name: str = Field(description="Canonical ENGLISH base name WITHOUT the equipment, e.g. 'Romanian Deadlift' or 'Single-Leg Romanian Deadlift'.")
    equipment: EquipmentType = Field(description="The equipment of this variant.")
    label_es: str = Field(description="Spanish label shown to the user, including the equipment, e.g. 'Peso muerto rumano (barra)'.")
    note_es: str = Field(description="One short Spanish sentence on what changes in this variant.")

    @property
    def slug(self) -> str:
        return make_slug(self.name, self.equipment)

    @property
    def full_name(self) -> str:
        return display_name(self.name, self.equipment)

class VariantOptions(BaseModel):
    interpreted_as_es: str = Field(description="Spanish sentence saying which exercise you understood from the request.")
    variants: list[ExerciseVariant] = Field(description="2 to 8 real, commonly programmed variants, most common first.")

SYSTEM_PROMPT = """You are a strength-and-conditioning librarian. The user types an exercise name, often in Spanish, possibly with typos or slang.

Return the concrete variants of THAT exercise that a coach would program:
- Different equipment (barbell, dumbbell, kettlebell, cable, machine, band, bodyweight) when the exercise is really done with it.
- Meaningful technique variants (single-leg/unilateral, deficit, sumo, close-grip, incline...) only when they are commonly used.
- The 'name' is the canonical English name used in exercise databases, WITHOUT the equipment (the equipment goes in its own field).
- Never invent exotic variants; never return two variants with the same name and equipment.
- Order from the most common to the least common. Between 2 and 8 variants.
- label_es and note_es are in Spanish; everything else in English.
"""

def _dedupe(options: VariantOptions) -> VariantOptions:
    seen, unique = set(), []
    for variant in options.variants:
        if variant.slug not in seen:
            seen.add(variant.slug)
            unique.append(variant)
    return options.model_copy(update={"variants": unique})

def resolve_variants(query: str) -> VariantOptions:
    """One Gemini call. Raises on API or schema errors (the CLI reports them)."""
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.1, max_retries=2)
    structured_llm = llm.with_structured_output(VariantOptions)
    options = structured_llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Request: {query}"),
    ])
    return _dedupe(options)
