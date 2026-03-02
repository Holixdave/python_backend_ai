import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Load dataset with question -> label
data = pd.read_csv("data.csv")

# data['question'] contains all ways of asking
# data['label'] contains the concept/intent

# Build a pipeline: text -> vector -> classifier
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(data['question'], data['label'])

# Save the trained model
joblib.dump(model, "ai_model.pkl")
print("Model trained and saved as ai_model.pkl")