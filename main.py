# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import csv
import random
from math_solver import solve_math_with_explanation
from equation_solver import solve_equation_with_steps

app = FastAPI(title="UTME26 AI Backend", description="Brilliant AI Study Assistant")

# Load model
qa_model = joblib.load("ai_model.pkl")

question_gen_model = None
if os.path.exists("question_gen_model.pkl"):
    question_gen_model = joblib.load("question_gen_model.pkl")

# Load label->answer mapping
label_to_answer: dict = {}

def load_answers():
    folder_path = "answers"
    if not os.path.exists(folder_path):
        print("Warning: answers folder not found.")
        return
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, newline='', encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        label = row.get("label", "").strip()
                        answer = row.get("answer", "").strip()
                        if label and answer:
                            label_to_answer.setdefault(label, []).append(answer)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    print(f"Loaded answers for {len(label_to_answer)} labels")

load_answers()


class QuestionRequest(BaseModel):
    query: str

class GenerateRequest(BaseModel):
    prompt: str


@app.post("/ai-query")
async def ask_ai(request: QuestionRequest):
    user_question = request.query.strip()

    # 1. Algebra
    eq_answer = solve_equation_with_steps(user_question)
    if eq_answer and "Could not" not in eq_answer:
        return {"label": "algebra", "answer": eq_answer}

    # 2. Arithmetic
    math_answer = solve_math_with_explanation(user_question)
    if math_answer:
        return {"label": "math", "answer": math_answer}

    # 3. ML model
    try:
        predicted_answer = qa_model.predict([user_question.lower()])[0]
        if predicted_answer and len(predicted_answer) > 5:
            label = "general"
            for lbl, answers in label_to_answer.items():
                if predicted_answer in answers:
                    label = lbl
                    break
            return {"label": label, "answer": predicted_answer}
    except Exception as e:
        print(f"Model prediction failed: {e}")

    # 4. Label fallback
    for lbl, answers in label_to_answer.items():
        if lbl in user_question.lower():
            return {"label": lbl, "answer": random.choice(answers)}

    return {"label": "unknown", "answer": "I'm still learning about that topic! Try asking about Math, Science, English, History, or Computer Science."}


@app.post("/generate-question")
async def generate_question(request: GenerateRequest):
    if question_gen_model is None:
        return {"error": "Question generation model not trained yet."}
    try:
        generated = question_gen_model.predict([request.prompt])[0]
        return {"question": generated}
    except Exception as e:
        return {"error": f"Generation failed: {e}"}


@app.get("/")
async def root():
    return {"message": "UTME26 AI backend is running and ready!",
            "endpoints": ["/ai-query", "/generate-question"]}


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": qa_model is not None}