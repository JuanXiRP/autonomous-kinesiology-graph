# main.py
# Interactive CLI: type an exercise, pick its variants, generate the chosen ones concurrently.
import logging
import os
import sys
import warnings

from dotenv import load_dotenv

# Suppress internal library warnings for a clean console output
warnings.filterwarnings("ignore")

# Load the environment before anything that reads it
load_dotenv()

from src.cli.prompts import choose_variants, confirm, print_options, print_summary
from src.infrastructure.db.exercise_repository import existing_slugs, similar_names
from src.pipeline.batch import run_variants
from src.resolution.variants import resolve_variants

REQUIRED_ENV = (
    "GOOGLE_API_KEY",
    "DATABASE_URL",
    "SUPABASE_URL",
    "SUPABASE_KEY",
)

def missing_env() -> list[str]:
    """Returns the names of required environment variables that are unset or empty."""
    return [name for name in REQUIRED_ENV if not os.environ.get(name)]

def interactive(query: str | None = None, input_fn=input) -> None:
    """Asks for an exercise, offers its variants, and generates the chosen ones concurrently."""
    missing = missing_env()
    if missing:
        print("❌ Missing required environment variables: " + ", ".join(missing))
        return

    query = (query or input_fn("¿Qué ejercicio quieres generar? ")).strip()
    if not query:
        print("Cancelado.")
        return

    print(f"🔎 Buscando variantes de '{query}'...")
    try:
        options = resolve_variants(query)
    except Exception as e:
        print(f"❌ No se pudieron obtener las variantes: {e}")
        return
    if not options.variants:
        print("❌ No se encontró ninguna variante para ese ejercicio.")
        return

    try:
        existing = existing_slugs([variant.slug for variant in options.variants])
        db_checked = True
    except Exception as e:
        print(f"⚠️ No se pudo consultar la base de datos: {e}")
        existing, db_checked = {}, False

    listed = {variant.full_name for variant in options.variants}
    related = [name for name in similar_names(options.variants[0].name) if name not in listed]

    print_options(options, existing, related)
    selected = choose_variants(options, existing, db_checked, input_fn)
    if not selected:
        print("Cancelado: no se generará nada.")
        return

    print("\nSe van a generar:")
    for variant in selected:
        action = "sobrescribir" if variant.slug in existing else "crear"
        print(f"  - {variant.label_es} ({variant.slug}) → {action}")
    print(f"Coste aproximado por variante: 2 llamadas a Gemini + ~3 min de render "
          f"(se ejecutan en paralelo, máximo 2 renders a la vez).")
    if not confirm("¿Generar?", input_fn):
        print("Cancelado.")
        return

    results = run_variants(selected, set(existing))
    print_summary(results)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(threadName)s | %(message)s")
    # Usage: python main.py                      -> asks for the exercise
    #        python main.py "peso muerto rumano" -> same, with the request prefilled
    interactive(" ".join(sys.argv[1:]) or None)
