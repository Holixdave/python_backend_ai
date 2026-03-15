# train_ai.py
import os
import random
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import wikipedia
from gpt2_test import ask_gpt2
from math_solver import solve_math_with_explanation
from equation_solver import solve_equation_with_steps

# -------------------------------------------------------
# Conversation state
# -------------------------------------------------------
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

# -------------------------------------------------------
# Load CSV files from answers/
# -------------------------------------------------------
answers_folder = "answers"
qa_frames = []

for file in os.listdir(answers_folder):
    if file.endswith(".csv"):
        path = os.path.join(answers_folder, file)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Warning: Failed to read {file}: {e}")
            continue

        if all(col in df.columns for col in ["question", "label", "answer"]):
            qa_frames.append(df[["question", "label", "answer"]])
            print(f"Loaded {len(df)} rows from {file}")
        elif all(col in df.columns for col in ["question", "label"]):
            df2 = df[["question", "label"]].copy()
            df2["answer"] = df2["label"]
            qa_frames.append(df2)
            print(f"Loaded {len(df)} rows from {file}")
        elif all(col in df.columns for col in ["prompt", "completion"]):
            df2 = pd.DataFrame({
                "question": df["prompt"],
                "label": "general",
                "answer": df["completion"]
            })
            qa_frames.append(df2)
            print(f"Loaded {len(df)} rows from {file}")
        else:
            print(f"Skipped {file}: columns found = {list(df.columns)}")

if not qa_frames:
    raise ValueError("No valid CSV files found in the answers/ folder!")

qa_data = pd.concat(qa_frames, ignore_index=True)
qa_data = qa_data.dropna(subset=["question", "answer"])
qa_data["question"] = qa_data["question"].str.lower().str.strip()
print(f"Total training rows: {len(qa_data)}")

# -------------------------------------------------------
# Train with TF-IDF + Logistic Regression (smarter than Naive Bayes)
# -------------------------------------------------------
qa_model = Pipeline([
    ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
    ('classifier', LogisticRegression(max_iter=1000, C=5.0))
])
qa_model.fit(qa_data['question'], qa_data['answer'])
print("Model training complete!")
joblib.dump(qa_model, "ai_model.pkl")
print("Model saved as ai_model.pkl")

# -------------------------------------------------------
# Build direct lookup dict for fast exact/fuzzy matching
# -------------------------------------------------------
qa_lookup = {}
for _, row in qa_data.iterrows():
    key = row['question'].lower().strip()
    qa_lookup[key] = row['answer']

# -------------------------------------------------------
# Greeting triggers
# -------------------------------------------------------
GREETING_TRIGGERS = [
    "hello", "hi", "hey", "howdy", "hiya", "yo", "sup", "wassup", "whats up",
    "what's up", "greetings", "salutations", "good morning", "good afternoon",
    "good evening", "good night", "good day", "morning", "evening", "afternoon",
    "how are you", "how do you do", "how's it going", "how is it going",
    "nice to meet you", "pleased to meet you", "i'm back", "i am back",
    "long time no see", "what is your name", "what's your name", "who are you",
    "what are you", "are you a robot", "are you human", "are you an ai",
    "can you help me", "i need help", "please help me", "help me",
    "thanks", "thank you", "thank you so much", "thanks a lot", "ok thanks",
    "bye", "goodbye", "see you", "see you later", "take care",
    "have a good day", "have a nice day", "good to see you",
    "what can you do", "what do you know", "are you smart", "are you intelligent",
    "can you solve math", "can you answer questions",
    "quiz me", "test me", "let's study", "let us study",
    "i'm ready", "ready", "start", "begin", "lol", "haha", "wow",
    "i'm bored", "i am bored", "okay", "ok", "alright", "cool", "awesome",
    "i want to learn", "teach me", "i want to study", "help me study",
    "i'm confused", "i am confused", "i don't understand", "i do not understand",
    "explain", "tell me about", "i failed", "i give up", "am i smart enough",
    "i want to pass", "i want to pass my exams", "i want to pass utme",
    "how do i study", "how do i study better"
]

def detect_intent(q_lower: str) -> str:
    stripped = q_lower.strip()
    for t in GREETING_TRIGGERS:
        if stripped == t or stripped.startswith(t + " ") or stripped.startswith(t + "!") or t in stripped:
            return "greeting"
    if any(w in q_lower for w in ["derivative", "integral", "calculus", "differentiate", "integrate"]):
        return "calculus"
    if any(w in q_lower for w in ["solve", "equation", "find x", "find y"]):
        return "algebra"
    if any(w in q_lower for w in ["plus", "minus", "times", "divide", "sum of",
                                    "subtract", "multiply", "squared", "square root",
                                    "cube of", "cube root", "% of", "percent of",
                                    "calculate", "what is 1", "what is 2", "what is 3",
                                    "what is 4", "what is 5", "what is 6",
                                    "what is 7", "what is 8", "what is 9"]):
        return "arithmetic"
    if any(w in q_lower for w in ["noun", "verb", "adjective", "adverb", "pronoun",
                                    "grammar", "tense", "simile", "metaphor",
                                    "synonym", "antonym", "clause", "punctuation",
                                    "essay", "paragraph", "comprehension"]):
        return "english"
    if any(w in q_lower for w in ["photosynthesis", "cell", "dna", "mitosis",
                                    "meiosis", "osmosis", "diffusion", "evolution",
                                    "respiration", "ecosystem", "food chain"]):
        return "biology"
    if any(w in q_lower for w in ["newton", "velocity", "acceleration", "force",
                                    "momentum", "kinetic", "potential energy",
                                    "gravity", "wave", "circuit", "power", "work"]):
        return "physics"
    if any(w in q_lower for w in ["atom", "element", "compound", "mixture",
                                    "ph", "acid", "base", "bond", "oxidation",
                                    "periodic table", "molecule", "reaction"]):
        return "chemistry"
    if any(w in q_lower for w in ["war", "history", "president", "independence",
                                    "revolution", "colonialism", "democracy", "empire"]):
        return "history"
    if any(w in q_lower for w in ["capital", "continent", "river", "country",
                                    "mountain", "ocean", "climate", "geography",
                                    "population", "map", "africa", "nigeria"]):
        return "geography"
    if any(w in q_lower for w in ["gdp", "inflation", "supply", "demand",
                                    "economics", "trade", "budget", "tax",
                                    "opportunity cost", "market"]):
        return "economics"
    if any(w in q_lower for w in ["python", "programming", "algorithm", "variable",
                                    "function", "loop", "database", "html", "css",
                                    "javascript", "computer", "software", "binary",
                                    "artificial intelligence", "machine learning",
                                    "network", "internet", "operating system"]):
        return "computer"
    return "general"

# -------------------------------------------------------
# Format math answers
# -------------------------------------------------------
def format_math_answer(result: str) -> str:
    prefixes = [
        "Here's the solution:",
        "Great question! Let me calculate that:",
        "Math solved!",
        "Here you go:",
    ]
    return f"{random.choice(prefixes)}\n{result}"

# -------------------------------------------------------
# Wikipedia fetch
# -------------------------------------------------------
def fetch_online_summary(query: str) -> str:
    try:
        summary = wikipedia.summary(query, sentences=3, auto_suggest=True)
        return summary
    except Exception:
        return None

# -------------------------------------------------------
# Extract subject from text
# -------------------------------------------------------
def extract_subject(text: str) -> str | None:
    capitalized = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
    if capitalized:
        return " ".join(capitalized[:2])
    words = text.strip().split()
    if words:
        return words[-1]
    return None

# -------------------------------------------------------
# Fuzzy match against lookup dict
# -------------------------------------------------------
def fuzzy_lookup(question: str) -> str | None:
    q = question.lower().strip()
    # Exact match
    if q in qa_lookup:
        return qa_lookup[q]
    # Partial match: question starts with a known key
    for key, answer in qa_lookup.items():
        if q.startswith(key) or key.startswith(q):
            return answer
    # Word overlap match
    q_words = set(q.split())
    best_score = 0
    best_answer = None
    for key, answer in qa_lookup.items():
        k_words = set(key.split())
        if not k_words:
            continue
        overlap = len(q_words & k_words) / len(k_words)
        if overlap > best_score and overlap >= 0.7:
            best_score = overlap
            best_answer = answer
    return best_answer

# -------------------------------------------------------
# Main ask function
# -------------------------------------------------------
def ask(question: str) -> str:
    q_lower = question.lower().strip()

    # Resolve pronouns using last subject
    last_subject = get_subject()
    if last_subject:
        for pronoun in ["he", "she", "it", "they", "his", "her", "their", "its"]:
            if re.search(r'\b' + pronoun + r'\b', q_lower):
                question = re.sub(r'\b' + pronoun + r'\b', last_subject,
                                  question, flags=re.IGNORECASE)
                q_lower = question.lower().strip()

    intent = detect_intent(q_lower)
    set_intent(intent)

    # --- Algebra ---
    if intent == "algebra":
        eq_ans = solve_equation_with_steps(question)
        if eq_ans and "Could not" not in eq_ans:
            add_history(question, eq_ans)
            return eq_ans

    # --- Arithmetic / Calculus ---
    if intent in ["arithmetic", "calculus"]:
        math_ans = solve_math_with_explanation(question)
        if math_ans:
            formatted = format_math_answer(math_ans)
            add_history(question, formatted)
            return formatted
        # --- Direct fuzzy lookup in CSV data ---
    direct = fuzzy_lookup(q_lower)
    if direct:
        add_history(question, direct)
        subj = extract_subject(direct)
        if subj:
            set_subject(subj)
        return direct

    # --- Label-based CSV match ---
    if intent in qa_data["label"].values:
        label_answers = qa_data[qa_data['label'] == intent]["answer"].tolist()
        if label_answers:
            selected = random.choice(label_answers)
            add_history(question, selected)
            subj = extract_subject(selected)
            if subj:
                set_subject(subj)
            return selected

    # --- Trained ML model ---
    try:
        predicted = qa_model.predict([q_lower])[0]
        if predicted and len(predicted) > 5:
            add_history(question, predicted)
            subj = extract_subject(predicted)
            if subj:
                set_subject(subj)
            return predicted
    except Exception:
        pass

    # --- Wikipedia fallback ---
    wiki = fetch_online_summary(question)
    if wiki:
        add_history(question, wiki)
        subj = extract_subject(wiki)
        if subj:
            set_subject(subj)
        return wiki

    # --- GPT-2 fallback ---
    try:
        gpt_answer = ask_gpt2(question)
        if gpt_answer and "unavailable" not in gpt_answer.lower() and len(gpt_answer) > 5:
            add_history(question, gpt_answer)
            return gpt_answer
    except Exception:
        pass

    return ("I don't have a specific answer for that yet, but I'm always learning! "
            "Try asking about Math, Science, English, History, Geography, Economics, or Computer Science.")


# -------------------------------------------------------
# Run interactive chat
# -------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  UTME26 AI — Your Brilliant Study Assistant")
    print("="*50)
    print("Type your question and press Enter.")
    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye! Keep studying hard. You've got this!")
            break
        print(f"\nUTME26 AI: {ask(user_input)}\n")