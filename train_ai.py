# train_ai.py
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
from math_solver import solve_math_with_explanation  # Import the math solver
import random
from equation_solver import solve_equation_with_steps
# -----------------------------
# Folder containing all CSV files
# -----------------------------
answers_folder = "answers"
qa_frames = []

# Load all CSVs in the folder
for file in os.listdir(answers_folder):
    if file.endswith(".csv"):
        path = os.path.join(answers_folder, file)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"⚠️ Failed to read {file}: {e}")
            continue

        # Determine format
        if all(col in df.columns for col in ["question", "label", "answer"]):
            # Full format
            qa_frames.append(df[["question", "label", "answer"]])
            print(f"✅ Loaded {len(df)} rows from {file} (question,label,answer format)")
        elif all(col in df.columns for col in ["question", "label"]):
            # Only question and label
            qa_frames.append(df[["question", "label"]].rename(columns={"label": "answer"}))
            print(f"✅ Loaded {len(df)} rows from {file} (question,label format)")
        elif all(col in df.columns for col in ["label", "answer"]):
            # label,answer format (we treat answer as input, label as answer)
            qa_frames.append(df[["answer", "label"]].rename(columns={"answer": "question", "label": "answer"}))
            print(f"✅ Loaded {len(df)} rows from {file} (label,answer format)")
        else:
            print(f"⚠️ {file} skipped: missing required columns")

# Check if we loaded any data
if not qa_frames:
    raise ValueError("No CSV files with the correct columns found!")

# Combine all CSVs
qa_data = pd.concat(qa_frames, ignore_index=True)
print(f"✅ Total loaded rows: {len(qa_data)}")

# -----------------------------
# Train QA model
# -----------------------------
qa_model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

qa_model.fit(qa_data['question'], qa_data['answer'])
print("✅ Model training complete!")

# Save the model
joblib.dump(qa_model, "ai_model.pkl")
print("✅ Model saved as ai_model.pkl")
def format_math_answer(expr_result: str) -> str:
    """
    Wraps the math solution with friendly text and optional emoji.
    """
    greetings = [
        "Here's what I got 🧮:", 
        "After some quick math ✨:", 
        "Let's solve that 📝:", 
        "Crunching numbers 🔢:"
    ]
    
    emojis = ["✅", "📊", "💡", "🧠", "🤓"]

    greeting = random.choice(greetings)
    emoji = random.choice(emojis)

    return f"{greeting} {expr_result} {emoji}"
conversation_state = {
    "last_intent": None
}
def ask(question: str) -> str:
    global conversation_state

    q_lower = question.lower()

    # -------------------------
    # Detect equation intent
    # -------------------------
    if "equation" in q_lower:
        conversation_state["last_intent"] = "solve_equation"
        return "Sure 😊 Please send me the equation you want to solve."

    # -------------------------
    # If last intent was equation
    # -------------------------
    if conversation_state["last_intent"] == "solve_equation":
        equation_answer = solve_equation_with_steps(question)
        if equation_answer:
            conversation_state["last_intent"] = None
            return equation_answer
        else:
            return "Hmm 🤔 That doesn't look like a valid equation. Try again."

    # -------------------------
    # Normal Equation Detection
    # -------------------------
    equation_answer = solve_equation_with_steps(question)
    if equation_answer is not None:
        return equation_answer

    # -------------------------
    # Arithmetic
    # -------------------------
    math_answer = solve_math_with_explanation(question)
    if math_answer is not None:
        return math_answer

    # -------------------------
    # Fallback to QA model
    # -------------------------
    return qa_model.predict([question])[0]

# -----------------------------
# Interactive testing
# -----------------------------
if __name__ == "__main__":
    while True:
        user_input = input("\nAsk a question (or type 'exit'): ").strip()
        if user_input.lower() == "exit":
            break
        print("Answer:", ask(user_input))