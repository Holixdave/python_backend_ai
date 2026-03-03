from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os
import csv
import random

app = FastAPI()

# ----------------------------
# Load Intent Classification Model
# ----------------------------
with open("intent_model.pkl", "rb") as f:
    intent_model = pickle.load(f)

# ----------------------------
# Load Question Generation Model (optional)
# ----------------------------
with open("question_gen_model.pkl", "rb") as f:
    question_gen_model = pickle.load(f)

# ----------------------------
# Load All Answers From /answers Folder
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
            with open(file_path, newline='', encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    label = row["label"].strip()
                    answer = row["answer"].strip()
                    label_to_answer[label] = answer

    print(f"✅ Loaded {len(label_to_answer)} answers.")

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
@app.post("/ask")
async def ask_ai(request: QuestionRequest):
    user_question = request.question.strip()

    try:
        predicted_label = intent_model.predict([user_question])[0]

        answer = label_to_answer.get(predicted_label)

        if answer:
            return {
                "label": predicted_label,
                "answer": answer
            }
        else:
            return {
                "label": predicted_label,
                "answer": "I'm not fully trained on this topic yet."
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