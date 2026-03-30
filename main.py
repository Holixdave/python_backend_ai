# main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import os
from math_solver import solve_math_with_explanation
from equation_solver import solve_equation_with_steps
from gpt2_test import ask_gpt2

app = FastAPI(
    title="UTME26 AI Backend",
    description="Brilliant AI Study Assistant"
)

# -------------------------------------------------------
# Pydantic models — DO NOT CHANGE (frontend depends on these)
# -------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
class ChatMessage(BaseModel):
    role: str # "user" or "assistant"
    content: str

class QuestionRequest(BaseModel):
    query: str
    history: List[ChatMessage] = Field(default_factory = list) # The secret to memory!

#--------------------------------------------------------
# -------------------------------------------------------
# Math detection helper
# -------------------------------------------------------
MATH_KEYWORDS = [
    "solve", "find x", "find y", "equation", "derivative", "integral",
    "calculus", "differentiate", "integrate", "plus", "minus", "times",
    "divide", "sum of", "subtract", "multiply", "squared", "square root",
    "cube of", "cube root", "% of", "percent of", "calculate",
]

def is_math_question(text: str) -> bool:
    t = text.lower()
    if any(kw in t for kw in MATH_KEYWORDS):
        return True
    # Detect expressions like "what is 5 + 3" or "2 * 9"
    import re
    if re.search(r'\d+\s*[\+\-\*\/\^]\s*\d+', t):
        return True
    return False


# -------------------------------------------------------
# Build a UTME-focused prompt so GPT-2 stays on topic
# -------------------------------------------------------
def build_prompt(user_query: str) -> str:
    return (
        "You are UTME26 AI, a brilliant Nigerian study assistant helping "
        "students prepare for JAMB UTME exams. You answer questions clearly "
        "and accurately about Math, English, Physics, Chemistry, Biology, "
        "History, Geography, Economics, and Computer Science.\n\n"
        f"Student: {user_query}\n"
        "UTME26 AI:"
    )


# -------------------------------------------------------
# POST /ai-query  — main chat endpoint (UNCHANGED for frontend)
# -------------------------------------------------------
@app.post("/ai-query")
async def ask_ai(request: QuestionRequest):
    user_question = request.query.strip()
    
    # Convert Pydantic history objects to simple dictionaries for the AI
    chat_history = [m.dict() for m in request.history] if request.history else []

    if not user_question:
        return {"label": "unknown", "answer": "Please type a question!"}

    # 1. Check for Math/Algebra first (Calculators don't need history)
    eq_answer = solve_equation_with_steps(user_question)
    if eq_answer and "Could not" not in eq_answer:
        return {"label": "algebra", "answer": eq_answer}

    # 2. GPT-2/Llama-3 — Now with History!
    # We no longer use build_prompt() because the Professor training 
    # inside gpt2_test.py is much more intelligent.
    gpt_answer = ask_gpt2(user_question, history=chat_history)

    return {"label": "general", "answer": gpt_answer}

# -------------------------------------------------------
# POST /generate-question  — UNCHANGED for frontend
# -------------------------------------------------------
@app.post("/generate-question")
async def generate_question(request: GenerateRequest):
    prompt = (
        f"Generate a JAMB UTME exam question about: {request.prompt}\n"
        "Question:"
    )
    try:
        result = ask_gpt2(prompt)
        if result and "unavailable" not in result.lower():
            # Clean up echo
            if "Question:" in result:
                result = result.split("Question:")[-1].strip()
            return {"question": result}
        return {"error": "Could not generate question. Try a more specific topic."}
    except Exception as e:
        return {"error": f"Generation failed: {e}"}


# -------------------------------------------------------
# GET /  — UNCHANGED
# -------------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "UTME26 AI backend is running and ready!",
        "endpoints": ["/ai-query", "/generate-question", "/health"]
    }


# -------------------------------------------------------
# GET /health  — UNCHANGED (used by loading screen)
# -------------------------------------------------------
@app.get("/health")
async def health():
    from gpt2_test import _pipe
    return {
        "status": "healthy",
        "model_loaded": _pipe is not None
    }