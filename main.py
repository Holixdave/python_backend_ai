# main.py
from fastapi import FastAPI
from pydantic import BaseModel
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
class QuestionRequest(BaseModel):
    query: str

class GenerateRequest(BaseModel):
    prompt: str

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

    if not user_question:
        return {"label": "unknown", "answer": "Please type a question!"}

    # 1. Equation solver (handles algebra like "solve 2x + 3 = 7")
    eq_answer = solve_equation_with_steps(user_question)
    if eq_answer and "Could not" not in eq_answer:
        return {"label": "algebra", "answer": eq_answer}

    # 2. Arithmetic solver (handles "what is 5 + 3", "square root of 144")
    if is_math_question(user_question):
        math_answer = solve_math_with_explanation(user_question)
        if math_answer:
            return {"label": "math", "answer": math_answer}

    # 3. GPT-2 — primary AI responder for everything else
    prompt = build_prompt(user_question)
    gpt_answer = ask_gpt2(prompt)

    # Clean up: remove the prompt echo if GPT-2 repeats it
    if "UTME26 AI:" in gpt_answer:
        gpt_answer = gpt_answer.split("UTME26 AI:")[-1].strip()
    if "Student:" in gpt_answer:
        gpt_answer = gpt_answer.split("Student:")[0].strip()

    if gpt_answer and "unavailable" not in gpt_answer.lower() and len(gpt_answer) > 5:
        return {"label": "general", "answer": gpt_answer}

    # 4. Final fallback — only if GPT-2 completely fails
    return {
        "label": "unknown",
        "answer": (
            "I'm still thinking about that one! Try asking about "
            "Math, Physics, Chemistry, Biology, English, History, "
            "Geography, Economics, or Computer Science."
        )
    }


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