from langgraph.graph import StateGraph, END
from typing import Literal

from src.domain.state import GraphState
from src.nodes.content_node import generate_exercise_content
from src.nodes.spec_node import generate_animation_spec
from src.nodes.render_node import render_exercise_animation
from src.nodes.storage_node import upload_exercise_media
from src.nodes.db_node import save_to_database

MAX_ATTEMPTS = 3

def route_after_generation(state: GraphState) -> Literal["content_generator", "spec_generator", "__end__"]:
    """
    Evaluates the state after content generation to determine the next step.
    Implements a self-healing retry loop pattern.
    """
    errors = state.get("validation_errors", [])
    attempts = state.get("generation_attempts", 0)

    # If there are errors and we have not exceeded the max retry limit, loop back
    if errors and attempts < MAX_ATTEMPTS:
        # If there's no structured exercise, it means the content node failed.
        if not state.get("structured_exercise"):
            return "content_generator"

    # If JSON was generated successfully, move to the animation spec node
    if state.get("structured_exercise"):
        return "spec_generator"

    # Max retries reached without any exercise data
    return "__end__"

def route_after_spec(state: GraphState) -> Literal["spec_generator", "render_generator", "storage_uploader"]:
    """
    Retries the animation spec while it keeps failing validation. When the retries are exhausted
    the exercise record is still stored, just without media.
    """
    if state.get("animation_spec"):
        return "render_generator"
    if state.get("spec_attempts", 0) < MAX_ATTEMPTS:
        return "spec_generator"
    return "storage_uploader"

# Initialize the state graph
workflow = StateGraph(GraphState)

# 1. Register all nodes
workflow.add_node("content_generator", generate_exercise_content)
workflow.add_node("spec_generator", generate_animation_spec)
workflow.add_node("render_generator", render_exercise_animation)
workflow.add_node("storage_uploader", upload_exercise_media)
workflow.add_node("db_writer", save_to_database)

# 2. Set entry point
workflow.set_entry_point("content_generator")

# 3. Define the edges
workflow.add_conditional_edges(
    "content_generator",
    route_after_generation,
    {
        "content_generator": "content_generator",
        "spec_generator": "spec_generator",
        "__end__": END
    }
)

workflow.add_conditional_edges(
    "spec_generator",
    route_after_spec,
    {
        "spec_generator": "spec_generator",
        "render_generator": "render_generator",
        "storage_uploader": "storage_uploader"
    }
)

# After the render, upload the media, then write to the database
workflow.add_edge("render_generator", "storage_uploader")
workflow.add_edge("storage_uploader", "db_writer")

# After database, the process is officially complete
workflow.add_edge("db_writer", END)
# Compile the workflow into a runnable LangChain Runnable instance
app = workflow.compile()
