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
        if "label" in df.columns and "answer" in df.columns:
            # Use 'answer' as input, 'label' as output
            qa_frames.append(df[["answer", "label"]].rename(columns={"answer": "question"}))

if not qa_frames:
    raise ValueError("No QA CSV files with 'label' and 'answer' found!")

qa_data = pd.concat(qa_frames, ignore_index=True)
print(f"✅ Loaded {len(qa_data)} QA pairs for training.")

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