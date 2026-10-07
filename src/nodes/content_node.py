# src/nodes/content_node.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from src.domain.models import ExerciseModel
from src.domain.state import GraphState

load_dotenv()

def generate_exercise_content(state: GraphState) -> dict:
    exercise_name = state.get("raw_exercise_name")
    attempts = state.get("generation_attempts", 0)
    
    # Using the stable 'flash' version to avoid 404 errors.
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.1,
        max_retries=2
    )
    
    structured_llm = llm.with_structured_output(ExerciseModel)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an elite kinesiologist, biomechanics expert, and software engineer.
        Your task is to analyze the requested exercise and provide a highly accurate, structured breakdown.
        
        CRITICAL INSTRUCTIONS:
        1. Always output strictly valid data matching the requested schema.
        2. The 'name' field MUST be the base name ONLY (e.g., "Bicep Curl"). Do NOT include the equipment.
        3. Do not mix execution steps with breathing cues. Keep them strictly in their respective fields.
        4. Focus on safety, joint mechanics, and optimal muscle contraction in 'common_mistakes'.
        """),
        ("human", "Analyze and generate structured data for this exercise: {exercise_name}")
    ])
    
    chain = prompt | structured_llm
    
    try:
        result = chain.invoke({"exercise_name": exercise_name})
        
        return {
            "structured_exercise": result,
            "generation_attempts": attempts + 1
        }
        
    except Exception as e:
        error_msg = f"Content generation failed on attempt {attempts + 1}: {str(e)}"
        
        return {
            "validation_errors": [error_msg],
            "generation_attempts": attempts + 1
        }