# test_node.py
import warnings
import os

# Suppress the LangChain AFC warnings to keep the terminal clean
warnings.filterwarnings("ignore")
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

from src.domain.state import GraphState
from src.nodes.content_node import generate_exercise_content

# Initialize a mock state
initial_state: GraphState = {
    "raw_exercise_name": "Incline Dumbbell Bench Press",
    "structured_exercise": None,
    "image_prompts": [],
    "validation_errors": [],
    "generation_attempts": 0
}

print("Iniciando generacion con Gemini...")
output = generate_exercise_content(initial_state)

if "structured_exercise" in output and output["structured_exercise"]:
    print("\nGeneracion exitosa:")
    print(output["structured_exercise"].model_dump_json(indent=2))
elif "validation_errors" in output:
    print(f"\nErrores detectados: {output['validation_errors']}")