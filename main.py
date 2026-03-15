# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import csv
from math_solver import solve_math_with_explanation
from equation_solver import solve_equation_with_steps

app = FastAPI(title="UTME26 AI Backend")

# ----------------------------
# Load Trained QA Model
# ----------------------------
qa_model = joblib.load("ai_model.pkl")

question_gen_model = None
if os.path.exists("question_gen_model.pkl"):
    question_gen_model = joblib.load("question_gen_model.pkl")

# ----------------------------
# Load Answers from CSVs by label
# ----------------------------
label_to_answer = {}


def load_answers():
    folder_path = "answers"
    if not os.path.exists(folder_path):
        print("⚠️ answers folder not found.")
        return

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, newline='', encoding="utf-8") as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        label = row.get("label", "").strip()
                        answer = row.get("answer", "").strip()
                        if label and answer:
                            if label in label_to_answer:
                                label_to_answer[label].append(answer)
                            else:
                                label_to_answer[label] = [answer]
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")

    print(f"✅ Loaded answers for {len(label_to_answer)} labels")


load_answers()


# ----------------------------
# Request Models
# ----------------------------
class QuestionRequest(BaseModel):
    query: str


class GenerateRequest(BaseModel):
    prompt: str


# ----------------------------
# AI Answer Endpoint
# ----------------------------
@app.post("/ai-query")
async def ask_ai(request: QuestionRequest):
    user_question = request.query.strip()

    # 1. Algebra solver
    eq_answer = solve_equation_with_steps(user_question)
    if eq_answer:
        return {"label": "algebra", "answer": eq_answer}

    # 2. Arithmetic solver
    math_answer = solve_math_with_explanation(user_question)
    if math_answer:
        return {"label": "math", "answer": math_answer}

    # 3. QA Model
    try:
        predicted_answer = qa_model.predict([user_question])[0]
        if predicted_answer:
            label = "general"
            for lbl, answers in label_to_answer.items():
                if predicted_answer in answers:
                    label = lbl
                    break
            return {"label": label, "answer": predicted_answer}
    except Exception as e:
        print(f"⚠️ QA model prediction failed: {e}")

    # 4. Fallback
    return {"label": "unknown", "answer": "I'm not fully trained on this topic yet."}


# ----------------------------
# Question Generation Endpoint
# ----------------------------
@app.post("/generate-question")
async def generate_question(request: GenerateRequest):
    if question_gen_model is None:
        return {"error": "Question generation model not trained."}
    try:
        generated = question_gen_model.predict([request.prompt])[0]
        return {"question": generated}
    except Exception as e:
        return {"error": f"Generation failed: {e}"}


# ----------------------------
# Health Check
# ----------------------------
@app.get("/")
async def root():
    return {"message": "UTME26 AI backend running!"}