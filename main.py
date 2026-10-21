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

from src.cli.prompts import ask_with_media, choose_variants, confirm, print_options, print_summary
from src.infrastructure.db.exercise_repository import existing_slugs, similar_names
from src.pipeline.batch import run_variants
from src.resolution.variants import resolve_variants

REQUIRED_ENV = (
    "GOOGLE_API_KEY",
    "DATABASE_URL",
)
# Only needed when the video and still are generated and uploaded
MEDIA_ENV = (
    "SUPABASE_URL",
    "SUPABASE_KEY",
)

TEXT_ONLY_FLAG = "--solo-texto"
WITH_MEDIA_FLAG = "--con-video"

def missing_env(with_media: bool = True) -> list[str]:
    """Returns the names of required environment variables that are unset or empty."""
    names = REQUIRED_ENV + MEDIA_ENV if with_media else REQUIRED_ENV
    return [name for name in names if not os.environ.get(name)]

def report_missing(missing: list[str]) -> None:
    print("❌ Missing required environment variables: " + ", ".join(missing))

def parse_args(argv: list[str]) -> tuple[str | None, bool | None]:
    """Splits the command line into the prefilled request and the media mode (None = ask)."""
    with_media = None
    words = []
    for arg in argv:
        if arg == TEXT_ONLY_FLAG:
            with_media = False
        elif arg == WITH_MEDIA_FLAG:
            with_media = True
        else:
            words.append(arg)
    return " ".join(words) or None, with_media

def interactive(query: str | None = None, input_fn=input, with_media: bool | None = None) -> None:
    """Asks for an exercise, offers its variants, and generates the chosen ones concurrently.
    with_media=None asks whether to render video and still; False stores only the text record."""
    missing = missing_env(with_media=bool(with_media))
    if missing:
        report_missing(missing)
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

    if with_media is None:
        with_media = ask_with_media(input_fn)
    if with_media:
        missing = missing_env(with_media=True)
        if missing:
            report_missing(missing)
            return

    print("\nSe van a generar" + ("" if with_media else " (solo texto, sin vídeo ni imagen)") + ":")
    for variant in selected:
        if variant.slug not in existing:
            action = "crear"
        elif with_media:
            action = "sobrescribir"
        else:
            action = "sobrescribir texto (conserva su vídeo/imagen)"
        print(f"  - {variant.label_es} ({variant.slug}) → {action}")
    if with_media:
        print("Coste aproximado por variante: 3+ llamadas a Gemini + ~4 min de render "
              "(se ejecutan en paralelo, máximo 2 renders a la vez).")
    else:
        print("Coste aproximado por variante: 1 llamada a Gemini, sin render.")
    if not confirm("¿Generar?", input_fn):
        print("Cancelado.")
        return

    results = run_variants(selected, set(existing), with_media=with_media)
    print_summary(results)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(threadName)s | %(message)s")
    # Usage: python main.py                                    -> asks for the exercise
    #        python main.py "peso muerto rumano"               -> same, with the request prefilled
    #        python main.py --solo-texto "peso muerto rumano"  -> text record only (media added later)
    #        python main.py --con-video "peso muerto rumano"   -> text + video + still, without asking
    request, media_mode = parse_args(sys.argv[1:])
    interactive(request, with_media=media_mode)
