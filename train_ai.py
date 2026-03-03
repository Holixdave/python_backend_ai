import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# ==========================
# 1️⃣ Training the QA model
# ==========================
# Load dataset with question -> label
qa_data = pd.read_csv("data.csv")  # existing file

# Build a pipeline: text -> vector -> classifier
qa_model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the QA model
qa_model.fit(qa_data['question'], qa_data['label'])

# Save the trained QA model
joblib.dump(qa_model, "ai_model.pkl")
print("QA Model trained and saved as ai_model.pkl")


# =================================
# 2️⃣ Training the Question Generator
# =================================
# Load dataset with prompt -> completion
gen_data = pd.read_csv("question_generation.csv")  # new CSV

# Simple example: train a Naive Bayes text model to predict completion
# (for small dataset; for production, consider GPT fine-tuning)
gen_model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the generation model
gen_model.fit(gen_data['prompt'], gen_data['completion'])

# Save the trained generation model
joblib.dump(gen_model, "question_gen_model.pkl")
print("Question Generation Model trained and saved as question_gen_model.pkl")