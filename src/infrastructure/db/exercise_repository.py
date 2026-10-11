# src/infrastructure/db/exercise_repository.py
# Read-only queries on the `exercises` table used by the interactive CLI.
import logging
import os
from datetime import datetime

import psycopg

def existing_slugs(slugs: list[str]) -> dict[str, datetime | None]:
    """Returns {slug: updated_at} for the slugs already stored. Raises on connection errors."""
    if not slugs:
        return {}
    with psycopg.connect(os.environ.get("DATABASE_URL")) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT slug, updated_at FROM exercises WHERE slug = ANY(%s)", (list(slugs),))
            return {slug: updated_at for slug, updated_at in cur.fetchall()}

def similar_names(term: str, limit: int = 10) -> list[str]:
    """Names of stored exercises containing `term` (case-insensitive). Empty list on any error."""
    term = term.strip()
    if not term:
        return []
    try:
        with psycopg.connect(os.environ.get("DATABASE_URL")) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT name FROM exercises WHERE name ILIKE %s ORDER BY name LIMIT %s",
                            (f"%{term}%", limit))
                return [row[0] for row in cur.fetchall()]
    except Exception as e:
        logging.warning(f"⚠️ [DB Agent] Could not search similar exercises: {e}")
        return []
