import os
import random
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import wikipedia
from gpt2_test import ask_gpt2
from math_solver import solve_math_with_explanation
from equation_solver import solve_equation_with_steps

# -----------------------------
# Conversation state (memory)
# -----------------------------
conversation_state = {
    "last_intent": None,
    "last_subject": None,
    "history": []
}

def set_intent(intent: str):
    conversation_state["last_intent"] = intent

def get_intent():
    return conversation_state.get("last_intent")

def set_subject(subject: str):
    conversation_state["last_subject"] = subject

def get_subject():
    return conversation_state.get("last_subject")

def add_history(question: str, answer: str):
    conversation_state["history"].append({"q": question, "a": answer})

# -----------------------------
# Load CSV files
# -----------------------------
answers_folder = "answers"
qa_frames = []

for file in os.listdir(answers_folder):
    if file.endswith(".csv"):
        path = os.path.join(answers_folder, file)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"⚠️ Failed to read {file}: {e}")
            continue

        if all(col in df.columns for col in ["question", "label", "answer"]):
            qa_frames.append(df[["question", "label", "answer"]])
            print(f"✅ Loaded {len(df)} rows from {file} (question,label,answer)")
        elif all(col in df.columns for col in ["question", "label"]):
            df2 = df[["question", "label"]].copy()
            df2["answer"] = df2["label"]
            qa_frames.append(df2)
            print(f"✅ Loaded {len(df)} rows from {file} (question,label)")
        # BUG FIX: question_generation.csv has prompt/completion columns — handle them
        elif all(col in df.columns for col in ["prompt", "completion"]):
            df2 = pd.DataFrame({
                "question": df["prompt"],
                "label": "general",
                "answer": df["completion"]
            })
            qa_frames.append(df2)
            print(f"✅ Loaded {len(df)} rows from {file} (prompt,completion)")
        else:
            print(f"⚠️ {file} skipped: missing required columns. Found: {list(df.columns)}")

if not qa_frames:
    raise ValueError("No CSV files with correct columns found!")

qa_data = pd.concat(qa_frames, ignore_index=True)
# Drop rows with missing question or answer
qa_data = qa_data.dropna(subset=["question", "answer"])
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
joblib.dump(qa_model, "ai_model.pkl")
print("✅ Model saved as ai_model.pkl")

# -----------------------------
# Format math answers
# -----------------------------
def format_math_answer(expr_result: str) -> str:
    greetings = [
        "Here's what I got 🧮:",
        "After some quick math ✨:",
        "Let's solve that 📝:",
        "Crunching numbers 🔢:"
    ]
    emojis = ["✅", "📊", "💡", "🧠", "🤓"]
    return f"{random.choice(greetings)} {expr_result} {random.choice(emojis)}"

# -----------------------------
# Wikipedia fetch
# -----------------------------
def fetch_online_summary(query: str) -> str:
    try:
        summary = wikipedia.summary(query, sentences=2, auto_suggest=True)
        return summary
    except Exception:
        return "I couldn't find anything online about that."

# -----------------------------
# Extract subject from text
# -----------------------------
def extract_subject(question: str) -> str:
    capitalized = re.findall(r'\b[A-Z][a-z]+\b', question)
    if capitalized:
        return " ".join(capitalized)
    words = question.strip().split()
    if words:
        return words[-1]
    return None

# -----------------------------
# Main ask function
# -----------------------------
def ask(question: str) -> str:
    q_lower = question.lower()

    # Resolve pronouns using last subject
    last_subject = get_subject()
    pronouns = ["he", "she", "it", "they", "his", "her", "their"]
    if last_subject:
        for pronoun in pronouns:
            if pronoun in q_lower.split():
                question = re.sub(r'\b' + pronoun + r'\b', last_subject, question, flags=re.IGNORECASE)
                q_lower = question.lower()

    # Intent detection
    if any(word in q_lower for word in ["derivative", "integral", "calculus"]):
        intent = "calculus"
    elif any(word in q_lower for word in ["solve", "equation"]):
        intent = "algebra"
    elif any(word in q_lower for word in ["plus", "minus", "times", "divide", "sum", "subtract", "multiply", "squared", "square root"]):
        intent = "arithmetic"
    elif any(word in q_lower for word in ["hello", "hi", "hey"]):
        intent = "greeting"
    elif any(word in q_lower for word in ["noun", "verb", "adjective", "grammar", "definition"]):
        intent = "grammar"
    else:
        intent = "general"

    set_intent(intent)

    # Algebra solver
    if intent == "algebra":
        eq_ans = solve_equation_with_steps(question)
        if eq_ans:
            add_history(question, eq_ans)
            subj = extract_subject(question)
            if subj:
                set_subject(subj)
            return eq_ans

    # Math/Calculus solver
    if intent in ["arithmetic", "calculus"]:
        math_ans = solve_math_with_explanation(question)
        if math_ans:
            formatted = format_math_answer(math_ans)
            add_history(question, formatted)
            return formatted

    # CSV label match
    if intent in qa_data["label"].values:
        answers = qa_data[qa_data['label'] == intent]["answer"].tolist()
        if answers:
            selected = random.choice(answers)
            add_history(question, selected)
            subj = extract_subject(selected)
            if subj:
                set_subject(subj)
            return selected

    # Trained QA model
    try:
        predicted = qa_model.predict([question])[0]
        add_history(question, predicted)
        subj = extract_subject(predicted)
        if subj:
            set_subject(subj)
        return predicted
    except Exception:
        pass

    # GPT fallback (requires OPENAI_API_KEY)
    try:
        gpt_answer = ask_gpt2(question)
        if gpt_answer and "failed" not in gpt_answer.lower():
            add_history(question, gpt_answer)
            subj = extract_subject(gpt_answer)
            if subj:
                set_subject(subj)
            return gpt_answer
    except Exception:
        pass

    # Wikipedia fallback
    online_answer = fetch_online_summary(question)
    add_history(question, online_answer)
    subj = extract_subject(online_answer)
    if subj:
        set_subject(subj)
    return online_answer  # BUG FIX: missing return statement in original


# BUG FIX: was "if name == 'main'" — missing underscores
if __name__ == "__main__":
    while True:
        user_input = input("\nAsk a question (or type 'exit'): ").strip()
        if user_input.lower() == "exit":
            break
        print("Answer:", ask(user_input))