from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import csv
import random

app = FastAPI(title="UTME26 AI Backend")

# ----------------------------
# Load Trained Models
# ----------------------------
intent_model = joblib.load("ai_model.pkl")
question_gen_model = None
if os.path.exists("question_gen_model.pkl"):
    question_gen_model = joblib.load("question_gen_model.pkl")

# ----------------------------
# Load Answers from /answers folder
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
                        label = row["label"].strip()
                        answer = row["answer"].strip()
                        if label in label_to_answer:
                            label_to_answer[label].append(answer)
                        else:
                            label_to_answer[label] = [answer]
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")

    print(f"✅ Loaded answers for {len(label_to_answer)} labels")

load_answers()

# ----------------------------
# Request Models (match Flutter)
# ----------------------------
class QuestionRequest(BaseModel):
    query: str  # matches Flutter's {"query": "..."} payload

class GenerateRequest(BaseModel):
    prompt: str

# ----------------------------
# AI Answer Endpoint
# ----------------------------
@app.post("/ai-query")
async def ask_ai(request: QuestionRequest):
    user_question = request.query.strip()
    try:
        predicted_label = intent_model.predict([user_question])[0]

        if predicted_label in label_to_answer:
            answer = random.choice(label_to_answer[predicted_label])
        else:
            answer = "I'm not fully trained on this topic yet."

        return {"label": predicted_label, "answer": answer}

    except Exception as e:
        return {"error": f"AI processing error: {e}"}

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