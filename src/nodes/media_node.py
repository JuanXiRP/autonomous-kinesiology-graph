import os
from google import genai
from google.genai import types

from src.domain.state import GraphState

def generate_exercise_media(state: GraphState) -> dict:
    """
    LangGraph node responsible for translating the structured exercise data
    into highly specific generative prompts and calling Google's Imagen 3 API.
    Maintains a strict visual aesthetic across all generated images.
    """
    exercise = state.get("structured_exercise")
    
    # Defensive programming: Ensure we have data before attempting generation
    if not exercise:
        return {"validation_errors": ["MediaNode failed: No structured exercise found in state."]}
        
    # Initialize the new Google GenAI client (bypassing the deprecated SDK)
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    
    # 1. Prompt Engineering for Visual Consistency (The "Art Direction")
    # We use the structured data to build a deterministic prompt.
    muscles = ", ".join(exercise.primary_muscles)
    base_prompt = (
        f"Minimalist 3D vector art style, clean white background. "
        f"A gender-neutral athletic stick-figure or mannequin performing the exercise: {exercise.name}. "
        f"The primary activated muscles ({muscles}) are glowing in a subtle neon blue highlight. "
        f"Action context: {exercise.preparation} "
        f"High quality, anatomical accuracy, fitness app aesthetic, isometric perspective, completely isolated on white."
    )
    
    try:
        # 2. Call Imagen 3 via the updated Google GenAI SDK
        # Using the standard Imagen 3 model alias
        result = client.models.generate_images(
            model='imagen-3.0-generate-001',
            prompt=base_prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                output_mime_type="image/jpeg",
                aspect_ratio="1:1" # Square aspect ratio is optimal for mobile UI
            )
        )
        
        # 3. File System Operations
        output_dir = "assets/images"
        os.makedirs(output_dir, exist_ok=True)
        
        image_path = f"{output_dir}/{exercise.slug}.jpg"
        
        # 4. Extract bytes and save to disk
        for generated_image in result.generated_images:
            with open(image_path, "wb") as f:
                f.write(generated_image.image.image_bytes)
            break # We only requested 1, but break just in case
            
        # 5. Mutate the Pydantic object to include the new media reference
        exercise.images = [image_path]
        
        return {
            "structured_exercise": exercise,
            "image_prompts": [base_prompt]
        }
        
    except Exception as e:
        # If image generation fails (e.g., safety filters triggered by anatomical terms),
        # we log the error but we DO NOT crash the pipeline. We return the JSON without the image.
        error_msg = f"Image generation skipped/failed: {str(e)}"
        return {"validation_errors": [error_msg]}