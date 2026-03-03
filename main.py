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
try:
    intent_model = joblib.load("ai_model.pkl")
    print("✅ Intent classification model loaded")
except Exception as e:
    print(f"❌ Failed to load ai_model.pkl: {e}")

try:
    question_gen_model = joblib.load("question_gen_model.pkl")
    print("✅ Question generation model loaded")
except Exception as e:
    print(f"❌ Failed to load question_gen_model.pkl: {e}")

# ----------------------------
# Load Answers from /answers folder
# ----------------------------
label_to_answer = {}

def load_answers():
    folder_path = "answers"

    if not os.path.exists(folder_path):
        print("⚠️ answers folder not found")
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
# Request Models
# ----------------------------
class QuestionRequest(BaseModel):
    question: str

class GenerateRequest(BaseModel):
    prompt: str

# ----------------------------
# AI Answer Endpoint
# ----------------------------
@app.post("/ai-query")
async def ask_ai(request: QuestionRequest):
    user_question = request.question.strip()
    try:
        predicted_label = intent_model.predict([user_question])[0]

        if predicted_label in label_to_answer:
            answer = random.choice(label_to_answer[predicted_label])
        else:
            answer = "I'm not fully trained on this topic yet."

        return {
            "label": predicted_label,
            "answer": answer
        }

    except Exception as e:
        return {"error": f"AI processing error: {e}"}

# ----------------------------
# Question Generation Endpoint
# ----------------------------
@app.post("/generate-question")
async def generate_question(request: GenerateRequest):
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