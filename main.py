from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import random

app = FastAPI()

# Load trained model
model = joblib.load("ai_model.pkl")

# Map labels to actual answers
label_to_answer = {
    "greeting": [
        "Hi there! How can I help you today?",
        "Hello! Nice to see you.",
        "Hey! How’s it going?",
        "Good day! How can I assist you?"
    ],
    "python": ["Python is a programming language used for coding."],
    "ai": ["AI stands for Artificial Intelligence, where machines simulate human intelligence."],
    "function": ["A function is a block of code that performs a specific task."],
    "list":[ "A list is a collection of items in a particular order, enclosed in square brackets."],
    "tuple":[ "A tuple is an immutable collection of items in Python."],
    "dictionary":[ "A dictionary stores key-value pairs for fast lookup."],
    "variable":[ "A variable stores data that can be used later in a program."],
    "string": ["A string is a sequence of characters enclosed in quotes."],
    "integer": ["An integer is a whole number without decimals."],
    "float":[ "A float is a number with decimal points."],
    "boolean":[ "A boolean is a data type that can be True or False."
],
     # Status
    "status": [
        "I'm just a program, but I'm running smoothly!",
        "Everything is fine here. How about you?",
        "All systems operational!",
        "I'm functioning as expected."
    ],

    # English Grammar
    "noun": [
        "A noun is a word used to identify a person, place, thing, or idea.",
        "Nouns name people, places, things, or concepts."
    ],
    "verb": [
        "A verb expresses an action or state of being.",
        "Verbs describe what happens or what someone does."
    ],
    "adjective": [
        "An adjective describes a noun or pronoun.",
        "Adjectives give more detail about a thing, person, or place."
    ],
    "adverb": [
        "An adverb modifies a verb, adjective, or another adverb.",
        "Adverbs explain how, when, where, or to what extent something happens."
    ],

    # Python Programming
    "python": [
        "Python is a versatile programming language used for many purposes including web development, data analysis, and AI."
    ],
    "function": [
        "A function is a block of code that performs a specific task.",
        "Functions help reuse code and organize your program."
    ],
    "list": [
        "A list is an ordered collection of items enclosed in square brackets.",
        "Lists can contain numbers, strings, or other objects."
    ],
    "tuple": [
        "A tuple is an immutable ordered collection of items.",
        "Tuples are similar to lists, but cannot be changed after creation."
    ],
    "dictionary": [
        "A dictionary stores key-value pairs for fast lookup.",
        "Dictionaries allow you to map keys to values."
    ],
    "variable": [
        "A variable stores data that can be used later in a program.",
        "Variables are used to hold values of different types."
    ],
    "string": [
        "A string is a sequence of characters enclosed in quotes.",
        "Strings can store text data in your program."
    ],
    "integer": [
        "An integer is a whole number without decimals."
    ],
    "float": [
        "A float is a number with decimal points."
    ],
    "boolean": [
        "A boolean is a data type that can be True or False."
    ],
    "python_loop": [
        "A loop allows you to execute code multiple times, like for or while loops."
    ],
    "python_if": [
        "An if statement lets your program make decisions based on conditions."
    ],
    "python_class": [
        "A class is a blueprint for creating objects in object-oriented programming."
    ],
    "python_set": [
        "A set is an unordered collection of unique items."
    ],
    "python_module": [
        "A module is a file containing Python code that can be imported into other programs."
    ],
    "python_file": [
        "A file in Python can be opened, read, and written using built-in functions."
    ],
    "python_exception": [
        "Exception handling allows your program to manage errors gracefully using try and except."
    ],
    "python_comment": [
        "Comments are used to explain code and are ignored by Python."
    ],
    "python_method": [
        "A method is a function associated with an object in Python."
    ],
    "python_operator": [
        "Operators in Python are symbols that perform operations on values, like +, -, *, /, %."
    ],
    "python_docstring": [
        "A docstring is a string literal used to document a Python function, class, or module."
    ],

    # Math
    "math_addition": [
        "Addition combines two or more numbers to get a total."
    ],
    "math_subtraction": [
        "Subtraction finds the difference between numbers."
    ],
    "math_multiplication": [
        "Multiplication combines numbers by repeated addition."
    ],
    "math_division": [
        "Division splits a number into equal parts."
    ],
    # Science
    "science_astronomy": [
        "Astronomy is the study of celestial bodies like stars, planets, and galaxies."
    ],
    "science_physics": [
        "Physics is the study of matter, energy, and the fundamental forces of nature."
    ],
    "science_chemistry": [
        "Chemistry studies the composition, properties, and reactions of substances."
    ],
    "science_biology": [
        "Biology is the study of living organisms and their interactions with the environment."
    ],

    # Geography
    "geography": [
        "Geography studies the Earth's landscapes, environments, and human interactions."
    ],

    # History
    "history": [
        "History is the study of past events and human societies."
    ],

    # AI & Machine Learning
    "ai": [
        "AI stands for Artificial Intelligence, where machines simulate human intelligence."
    ],
    "machine_learning": [
        "Machine learning allows computers to learn patterns from data and make predictions."
    ],

    # Computer Science
    "computer_science": [
        "Computer science studies algorithms, data structures, and programming to solve problems."
    ],

    # Web & Networking
    "web_development": [
        "Web development involves creating websites using HTML, CSS, and JavaScript."
    ],
    "networking": [
        "Networking deals with connecting computers and devices to share data and resources."
    ],
}


# Input models
class Message(BaseModel):
    text: str

class AIQuery(BaseModel):
    query: str

# Routes
@app.post("/echo")
def echo_message(msg: Message):
    return {"you_sent": msg.text}
@app.get("/")
def home():
    return {"message": "AI Backend is running successfully 🚀"}
# <-- REPLACE your old /ai-query with this one
@app.post("/ai-query")
async def ai_query(request: AIQuery):
    user_query = request.query.lower().strip()

    # Check if greeting first
    if any(word in user_query for word in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]):
        response = random.choice(label_to_answer["greeting"])
    elif any(word in user_query for word in ["how are you", "how’s it going", "what’s up", "how are things"]):
        response = random.choice(label_to_answer["status"])
    else:
        # Predict label using trained model
        predicted_label = model.predict([user_query])[0]
        responses = label_to_answer.get(predicted_label)
        if responses:
            if isinstance(responses, list):
                response = random.choice(responses)  # pick random if multiple responses
            else:
                response = responses
        else:
            response = "Sorry, I don't understand that yet."

    return {"answer": response}