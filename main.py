from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load trained model
model = joblib.load("ai_model.pkl")

# Map labels to actual answers
label_to_answer = {
    "greeting": "Hello! How can I help you today?",
    "status": "I'm just a program, but I'm working fine!",
    "python": "Python is a programming language used for coding.",
    "ai": "AI stands for Artificial Intelligence, where machines simulate human intelligence.",
    "function": "A function is a block of code that performs a specific task.",
    "list": "A list is a collection of items in a particular order, enclosed in square brackets.",
    "tuple": "A tuple is an immutable collection of items in Python.",
    "dictionary": "A dictionary stores key-value pairs for fast lookup.",
    "variable": "A variable stores data that can be used later in a program.",
    "string": "A string is a sequence of characters enclosed in quotes.",
    "integer": "An integer is a whole number without decimals.",
    "float": "A float is a number with decimal points.",
    "boolean": "A boolean is a data type that can be True or False."
}

# Input models
class Message(BaseModel):
    text: str

class AIQuery(BaseModel):
    query: str

# Routes
@app.get("/")
def read_root():
    return {"message": "Hello, AI Backend is running!"}

@app.post("/echo")
def echo_message(msg: Message):
    return {"you_sent": msg.text}

@app.post("/ai-query")
async def ai_query(request: AIQuery):
    user_query = request.query
    predicted_label = model.predict([user_query])[0]
    response = label_to_answer.get(predicted_label, "Sorry, I don't understand that yet.")
    return {"answer": response}