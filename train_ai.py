import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

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

        # Check columns
        if all(col in df.columns for col in ["question", "label", "answer"]):
            qa_frames.append(df[["question", "label", "answer"]])
            print(f"✅ Loaded {len(df)} rows from {file}")
        else:
            print(f"⚠️ {file} skipped: missing required columns ['question','label','answer']")

# Check if we loaded any data
if not qa_frames:
    raise ValueError("No CSV files with required columns ['question','label','answer'] found!")

# Combine all CSVs
qa_data = pd.concat(qa_frames, ignore_index=True)
print(f"✅ Total loaded rows: {len(qa_data)}")

# -----------------------------
# Train QA model
# -----------------------------
# We'll use question → answer mapping
qa_model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

qa_model.fit(qa_data['question'], qa_data['answer'])
print("✅ Model training complete!")

# Save the model
joblib.dump(qa_model, "ai_model.pkl")
print("✅ Model saved as ai_model.pkl")

# -----------------------------
# Optional: Quick test
# -----------------------------
def ask(question):
    answer = qa_model.predict([question])[0]
    return answer

if __name__ == "__main__":
    while True:
        user_input = input("\nAsk a question (or type 'exit'): ").strip()
        if user_input.lower() == "exit":
            break
        print("Answer:", ask(user_input))