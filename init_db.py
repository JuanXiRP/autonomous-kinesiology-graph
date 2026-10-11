# init_db.py
import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

def initialize_database():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ Error: DATABASE_URL not found in .env")
        return

    print("Connecting to Supabase...")
    
    # DDL: Data Definition Language
    create_table_query = """
    CREATE TABLE IF NOT EXISTS exercises (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        slug VARCHAR(255) UNIQUE NOT NULL,
        name VARCHAR(255) NOT NULL,
        force VARCHAR(50) NOT NULL,
        mechanic VARCHAR(50) NOT NULL,
        equipment VARCHAR(50) NOT NULL,
        category VARCHAR(50) NOT NULL,
        primary_muscles TEXT[] NOT NULL,
        secondary_muscles TEXT[] DEFAULT '{}',
        preparation TEXT NOT NULL,
        execution_steps TEXT[] NOT NULL,
        breathing TEXT NOT NULL,
        common_mistakes TEXT[] DEFAULT '{}',
        images TEXT[] DEFAULT '{}',
        animation_video VARCHAR(500),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """

    # CREATE TABLE IF NOT EXISTS does not alter an existing table
    add_updated_at_query = """
    ALTER TABLE exercises
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP;
    """

    try:
        # Context managers close the connection automatically
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_query)
                cur.execute(add_updated_at_query)
            conn.commit()
        print("✅ Table 'exercises' verified/created successfully in Supabase.")
    except Exception as e:
        print(f"❌ Error creating the table: {e}")

if __name__ == "__main__":
    initialize_database()