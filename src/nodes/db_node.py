import os
import psycopg
from src.domain.state import GraphState

def save_to_database(state: GraphState) -> dict:
    """
    Saves the fully processed ExerciseModel into PostgreSQL (Supabase) securely.
    """
    exercise = state.get("structured_exercise")
    
    if not exercise:
        return {"validation_errors": ["DBNode failed: No exercise data to save."]}
        
    db_url = os.environ.get("DATABASE_URL")
    
    try:
        print(f"\n[DB Agent] 💾 Saving '{exercise.slug}' to Supabase...")
        
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                # Upsert logic (Insert or Update if exists)
                query = """
                    INSERT INTO exercises (
                        slug, name, force, mechanic, equipment, category,
                        primary_muscles, secondary_muscles,
                        preparation, execution_steps, breathing, common_mistakes,
                        images, animation_video
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (slug) DO UPDATE SET
                        name = EXCLUDED.name,
                        force = EXCLUDED.force,
                        mechanic = EXCLUDED.mechanic,
                        equipment = EXCLUDED.equipment,
                        category = EXCLUDED.category,
                        primary_muscles = EXCLUDED.primary_muscles,
                        secondary_muscles = EXCLUDED.secondary_muscles,
                        preparation = EXCLUDED.preparation,
                        execution_steps = EXCLUDED.execution_steps,
                        breathing = EXCLUDED.breathing,
                        common_mistakes = EXCLUDED.common_mistakes,
                        images = CASE WHEN cardinality(EXCLUDED.images) > 0 THEN EXCLUDED.images ELSE exercises.images END,
                        animation_video = COALESCE(EXCLUDED.animation_video, exercises.animation_video),
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id;
                """
                
                # Extract values from Pydantic model
                values = (
                    exercise.slug, exercise.name, exercise.force.value,
                    exercise.mechanic.value, exercise.equipment.value, exercise.category.value,
                    exercise.primary_muscles, exercise.secondary_muscles,
                    exercise.preparation, exercise.execution_steps, exercise.breathing,
                    exercise.common_mistakes, exercise.images, exercise.animation_video
                )
                
                cur.execute(query, values)
                inserted_id = cur.fetchone()[0]
            
            # Commit the transaction
            conn.commit()
            
        print(f"[DB Agent] ✅ Transaction completed! DB_ID: {inserted_id}\n")
        
        return {} 
        
    except Exception as e:
        error_msg = f"Database insertion failed: {str(e)}"
        print(f"\n❌ [DB Agent] CRITICAL ERROR: {error_msg}\n")
        return {"validation_errors": [error_msg]}