# main.py
import warnings
import os

# Suppress internal warnings for a clean console output
warnings.filterwarnings("ignore")
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

from src.graph import app

def run_pipeline(exercise_name: str):
    """
    Executes the LangGraph ETL pipeline and streams the state changes.
    """
    print(f"🚀 Starting Generative ETL Pipeline for: '{exercise_name}'\n")
    
    initial_state = {
        "raw_exercise_name": exercise_name,
        "structured_exercise": None,
        "image_prompts": [],
        "validation_errors": [],
        "generation_attempts": 0
    }
    
    # Run the compiled graph and stream the execution steps
    final_state = app.invoke(initial_state)
    
    if final_state.get("structured_exercise"):
        print("✅ Pipeline Completed Successfully. Final JSON:")
        print(final_state["structured_exercise"].model_dump_json(indent=2))
    else:
        print("❌ Pipeline failed to generate valid data after max retries.")
        print(f"Errors: {final_state.get('validation_errors')}")

if __name__ == "__main__":
    # Test with a compound exercise
    run_pipeline("Romanian Deadlift")