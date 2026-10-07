from langgraph.graph import StateGraph, END
from typing import Literal

from src.domain.state import GraphState
from src.nodes.content_node import generate_exercise_content
from src.nodes.media_node import generate_exercise_media
from src.nodes.video_node import generate_video_script
from src.nodes.db_node import save_to_database

def route_after_generation(state: GraphState) -> Literal["content_generator", "__end__"]:
    """
    Evaluates the state after content generation to determine the next step.
    Implements a self-healing retry loop pattern.
    """
    errors = state.get("validation_errors", [])
    attempts = state.get("generation_attempts", 0)
    
    # If there are errors and we have not exceeded the max retry limit, loop back
    if errors and attempts < 3:
        # We check if the error came from the media node or content node.
        # If there's no structured exercise, it means the content node failed.
        if not state.get("structured_exercise"):
            return "content_generator"
        
    # If JSON was generated successfully, move to the Image Generation node
    if state.get("structured_exercise") and not state.get("image_prompts"):
        return "media_generator"
        
    # If everything is done (or max retries reached), exit the graph
    return "__end__"
# Initialize the state graph
workflow = StateGraph(GraphState)

# 1. Register all nodes
workflow.add_node("content_generator", generate_exercise_content)
workflow.add_node("media_generator", generate_exercise_media)
workflow.add_node("video_generator", generate_video_script)
workflow.add_node("db_writer", save_to_database) # <-- REGISTER DB NODE

# 2. Set entry point
workflow.set_entry_point("content_generator")

# 3. Define the sequential edges
workflow.add_conditional_edges(
    "content_generator",
    route_after_generation,
    {
        "content_generator": "content_generator", 
        "media_generator": "media_generator",     
        "__end__": END                            
    }
)

# After media, go to video
workflow.add_edge("media_generator", "video_generator")

# After video, go to the database
workflow.add_edge("video_generator", "db_writer")

# After database, the process is officially complete
workflow.add_edge("db_writer", END)
# Compile the workflow into a runnable LangChain Runnable instance
app = workflow.compile()