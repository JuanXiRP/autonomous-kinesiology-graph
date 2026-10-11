# src/cli/prompts.py
# Terminal dialogue (in Spanish) for picking exercise variants. Input/print are injectable for tests.
from datetime import datetime

from src.pipeline.batch import VariantResult
from src.resolution.variants import ExerciseVariant, VariantOptions

YES = {"s", "si", "sí", "y", "yes"}
ALL = {"todos", "todas", "all", "*"}

STATUS_ES = {
    "created": "✅ creado",
    "updated": "♻️ actualizado",
    "no_media": "⚠️ guardado sin vídeo",
    "failed": "❌ falló",
}

def parse_selection(text: str, count: int) -> list[int] | None:
    """'1,3', '1-3', '2 4', 'todos' -> zero-based indices; '' -> [] (cancel); invalid -> None."""
    text = text.strip().lower()
    if not text:
        return []
    if text in ALL:
        return list(range(count))
    indices: list[int] = []
    for token in text.replace(" ", ",").split(","):
        if not token:
            continue
        try:
            if "-" in token:
                first, last = (int(part) for part in token.split("-", 1))
                if first > last:
                    return None
                numbers = range(first, last + 1)
            else:
                numbers = [int(token)]
        except ValueError:
            return None
        for number in numbers:
            if not 1 <= number <= count:
                return None
            if number - 1 not in indices:
                indices.append(number - 1)
    return indices

def confirm(question: str, input_fn=input) -> bool:
    return input_fn(f"{question} (s/N): ").strip().lower() in YES

def format_date(value: datetime | None) -> str:
    return value.strftime("%d/%m/%Y") if value else "fecha desconocida"

def print_options(options: VariantOptions, existing: dict, related: list[str], print_fn=print) -> None:
    print_fn(f"\n🔎 {options.interpreted_as_es}\n")
    for number, variant in enumerate(options.variants, start=1):
        mark = f"  [ya en BD, {format_date(existing[variant.slug])}]" if variant.slug in existing else ""
        print_fn(f"  {number}) {variant.label_es} — {variant.slug}{mark}")
        print_fn(f"       {variant.note_es}")
    if related:
        print_fn("\n  Ejercicios relacionados que ya están en la BD: " + ", ".join(related))

def choose_variants(options: VariantOptions, existing: dict, db_checked: bool,
                    input_fn=input, print_fn=print) -> list[ExerciseVariant]:
    """Asks which variants to generate; variants already stored need an explicit overwrite confirmation."""
    count = len(options.variants)
    while True:
        answer = input_fn(f"\n¿Cuáles quieres generar? (ej. 1,3 · 1-{count} · todos · Enter para cancelar): ")
        indices = parse_selection(answer, count)
        if indices is not None:
            break
        print_fn(f"  Selección no válida. Usa números del 1 al {count}.")

    selected = []
    for index in indices:
        variant = options.variants[index]
        if variant.slug in existing:
            if not confirm(f"  '{variant.label_es}' ya existe. ¿Regenerarlo y sobrescribirlo?", input_fn):
                print_fn(f"  Se omite {variant.slug}.")
                continue
        selected.append(variant)

    if selected and not db_checked:
        print_fn("\n⚠️ No se pudo comprobar la base de datos: alguna variante podría sobrescribir un ejercicio existente.")
        if not confirm("¿Continuar igualmente?", input_fn):
            return []
    return selected

def print_summary(results: list[VariantResult], print_fn=print) -> None:
    print_fn("\n📋 Resumen")
    for result in results:
        print_fn(f"\n  {STATUS_ES[result.status]} · {result.variant.label_es} ({result.variant.slug})")
        if result.video_url:
            print_fn(f"     Vídeo:  {result.video_url}")
        for url in result.image_urls:
            print_fn(f"     Imagen: {url}")
        for warning in result.warnings:
            print_fn(f"     ⚠️ {warning[:300]}")
