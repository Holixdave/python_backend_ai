import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# -----------------------------
# Load all CSVs from /answers folder
# -----------------------------
answers_folder = "answers"
qa_frames = []

for file in os.listdir(answers_folder):
    if file.endswith(".csv"):
        path = os.path.join(answers_folder, file)
        df = pd.read_csv(path)
        if "question" in df.columns and "label" in df.columns:
            qa_frames.append(df[["question", "label"]])

if not qa_frames:
    raise ValueError("No QA CSV files with 'question' and 'label' found!")

qa_data = pd.concat(qa_frames, ignore_index=True)
print(f"✅ Loaded {len(qa_data)} question-label pairs for training.")

# -----------------------------
# Train QA model
# -----------------------------
qa_model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])
qa_model.fit(qa_data['question'], qa_data['label'])
joblib.dump(qa_model, "ai_model.pkl")
print("QA Model trained and saved as ai_model.pkl")

# -----------------------------
# Train Question Generation Model (optional)
# -----------------------------
gen_file = "question_generation.csv"
if os.path.exists(gen_file):
    gen_data = pd.read_csv(gen_file)
    if "prompt" in gen_data.columns and "completion" in gen_data.columns:
        gen_model = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('classifier', MultinomialNB())
        ])
        gen_model.fit(gen_data['prompt'], gen_data['completion'])
        joblib.dump(gen_model, "question_gen_model.pkl")
        print("Question Generation Model trained and saved as question_gen_model.pkl")
else:
    print("⚠️ No question_generation.csv found. Skipping generation model.")