# src/nodes/video_node.py
import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.domain.state import GraphState

def generate_video_script(state: GraphState) -> dict:
    """
    Acts as a Senior Python Developer & Art Director. 
    Writes a Manim script with a fixed side text panel and a muscular stick figure.
    """
    exercise = state.get("structured_exercise")
    
    if not exercise:
        return {"validation_errors": ["VideoNode failed: No exercise data."]}
        
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.2
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Python developer and Art Director specialized in the 'manim' library.
        Your task is to generate a complete, runnable Python script using Manim.
        
        CRITICAL DESIGN & BRANDING RULES (MUST FOLLOW STRICTLY):
        1. **Colors & Background**: 
           - Scene background: `self.camera.background_color = "#F8FAF9"`
           - Main structures, bones, and text: Secondary Dark Green (`color="#1B4332"`).
           - Muscular highlights & active muscle groups ({muscles}): Primary Green (`color="#2D6A4F"`).
           - Equipment (barbell/weights): Dark Grey (`color="#444444"`).
        2. **Layout & UI (Side Panel)**:
           - Place all execution steps as a vertical bulleted list on the **LEFT side** of the screen. This text panel MUST remain **always visible** from start to finish.
           - Place the Exercise Name as a bold title at the top left/center.
           - Reserve the right side of the screen exclusively for the animated exercise figure.
        3. **Muscular Figure Design**:
           - Do NOT use thin single-pixel lines for the stickman. Represent limbs, torso, and shoulders using thick blocks, wide lines (`stroke_width=12` or higher), or filled shapes to give it a robust, muscular appearance.
        4. **Animation & Mechanics**:
           - Use standard Manim rate functions like `rate_func=smooth`. NEVER use `sine_ops`.
           - Class name must be `ExerciseAnimation(Scene)`. Output ONLY valid Python code. No markdown formatting (do not use ```python).
        """),
        ("human", "Exercise: {name}\nPreparation: {prep}\nSteps: {steps}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        raw_code = chain.invoke({
            "name": exercise.name,
            "muscles": ", ".join(exercise.primary_muscles),
            "prep": exercise.preparation,
            "steps": " \n".join([f"{i+1}. {step}" for i, step in enumerate(exercise.execution_steps)])
        })
        
        clean_code = re.sub(r"^```python\n|```$", "", raw_code, flags=re.MULTILINE).strip()
        
        output_dir = "assets/animations"
        os.makedirs(output_dir, exist_ok=True)
        
        script_path = f"{output_dir}/{exercise.slug}_animation.py"
        
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(clean_code)
            
        exercise.animation_video = script_path
        
        return {"structured_exercise": exercise}
        
    except Exception as e:
        error_msg = f"Video script generation failed: {str(e)}"
        return {"validation_errors": [error_msg]}