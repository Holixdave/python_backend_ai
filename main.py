from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import random
from sympy import symbols, Eq, solve, sympify

app = FastAPI()

# Load trained model
model = joblib.load("ai_model.pkl")

def solve_math_safe(question):
    x = symbols('x')
    
    if "solve" in question.lower():
        if "=" in question:
            lhs, rhs = question.split("=")
            try:
                eq = Eq(sympify(lhs.strip()), sympify(rhs.strip()))
                solution = solve(eq, x)
                return f"Solution: {solution}"
            except Exception as e:
                return f"Error parsing equation: {e}"
        else:
            return "Equation must contain '='"
    
    return "Not a solvable math question"
# Map labels to actual answers
label_to_answer = {
    "greeting": [
        "Hi there! How can I help you today?",
        "Hello! Nice to see you.",
        "Hey! How’s it going?",
        "Good day! How can I assist you?"
         "Hello! How can I help you today?",
        "Hi there! Ready to learn something new?",
        "Hey! What topic do you want to explore today?",
        "Good day! How can I assist you?",
        "Hello! Hope you’re having a great day!"
    ],
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
    "status": [
        "I’m just a program, but I’m running smoothly!",
        "All systems operational! How can I help?",
        "I’m here and ready to answer your questions!",
        "Everything is working fine on my side!"
    ],
    "python": [
        "Python is a high-level programming language used for coding.",
        "Python is versatile and widely used for web, data, AI, and scripting.",
        "Python was created by Guido van Rossum and emphasizes readability.",
        "Python lets you write clear and concise code quickly."
        "Python is a versatile programming language used for many purposes including web development, data analysis, and AI."
    ],
    "function": [
        "A function is a block of code that performs a specific task.",
        "Functions help reuse code and organize logic in Python.",
        "You can define a function in Python using the 'def' keyword.",
        "Functions may take input arguments and return results."
    ],
    "list": [
        "A list is an ordered collection of items in Python, enclosed in brackets [].",
        "Lists allow you to store multiple values together and access them by index.",
        "You can add, remove, or modify items in a Python list.",
        "Lists can contain elements of different data types."
    ],
    "tuple": [
        "A tuple is an ordered and immutable collection of items in Python.",
        "Tuples are defined using parentheses () and cannot be changed after creation.",
        "Tuples are useful for fixed collections of data.",
        "You can access tuple elements by index, just like lists."
    ],
    "dictionary": [
        "A dictionary stores key-value pairs in Python.",
        "Dictionaries are defined using curly braces {} with keys and values.",
        "You can access values by their keys in a dictionary.",
        "Dictionaries are unordered but allow fast lookup using keys."
    ],
    "variable": [
        "A variable stores data that can be used later in a program.",
        "Variables can hold numbers, strings, lists, or other types of data.",
        "You can assign a value to a variable using the '=' operator.",
        "Variable names should be descriptive and follow Python naming rules."
    ],
    "string": [
        "A string is a sequence of characters enclosed in quotes.",
        "Strings can store text, sentences, or any sequence of characters.",
        "You can manipulate strings using methods like .upper(), .lower(), and .split().",
        "Strings are immutable in Python, meaning they cannot be changed directly."
    ],
    "integer": [
        "An integer is a whole number without decimals.",
        "Integers can be positive, negative, or zero.",
        "You can perform arithmetic operations like addition, subtraction, and multiplication with integers.",
        "Integers are one of Python's basic numeric types."
    ],
    "float": [
        "A float is a number with decimal points.",
        "Floats can represent fractional values like 3.14 or 0.001.",
        "You can perform arithmetic operations with floats, including division.",
        "Python automatically treats numbers with decimals as floats."
    ],
    "boolean": [
        "A boolean is a data type that can be True or False.",
        "Booleans are often used for conditions and logical comparisons.",
        "In Python, comparisons like 5 > 3 return a boolean value.",
        "Boolean values can control the flow of programs using if statements."
    ],
    "loop": [
        "A loop allows you to repeat a block of code multiple times.",
        "Python has 'for' loops for iterating over sequences and 'while' loops for conditions.",
        "Loops help automate repetitive tasks in code.",
        "You can control loops using break, continue, and pass statements."
    ],
    "condition": [
        "An if statement lets you execute code only if a condition is true.",
        "You can use elif and else to handle multiple conditions.",
        "Conditions use comparison operators like ==, !=, <, >, <=, >=.",
        "Conditional statements help programs make decisions."
    ],
    "oop": [
        "Object-Oriented Programming (OOP) is a programming paradigm using classes and objects.",
        "A class defines a blueprint for objects, and objects are instances of classes.",
        "OOP concepts include inheritance, encapsulation, and polymorphism.",
        "OOP helps organize complex programs and promote code reuse."
    ],
    "module": [
        "A module is a Python file containing functions, classes, or variables.",
        "You can import modules into your program using the 'import' keyword.",
        "Packages are collections of modules organized in directories.",
        "Modules help organize code into reusable pieces."
    ],
    "error": [
        "An exception is an error that occurs during program execution.",
        "Python raises exceptions for problems like division by zero or invalid input.",
        "You can handle exceptions using try and except blocks.",
        "Debugging helps find and fix errors in your code."
    ],
    "algorithm": [
        "An algorithm is a step-by-step procedure to solve a problem.",
        "Recursion is when a function calls itself to solve smaller parts of a problem.",
        "Algorithms are essential for computer science and programming.",
        "Efficient algorithms help programs run faster and use less memory."
    ],
    "data_structure": [
        "Data structures store and organize data for efficient access and modification.",
        "Stacks follow LIFO (last-in, first-out) order, while queues follow FIFO (first-in, first-out).",
        "Linked lists are sequences of nodes connected by pointers.",
        "Sets store unique items without order, and dictionaries store key-value pairs."
    ],
    "web": [
        "APIs allow programs to communicate with each other over the web.",
        "HTTP is the protocol for transferring data on the web.",
        "HTML is used to structure web pages, and CSS styles them.",
        "JavaScript adds interactivity to web pages.",
        "JSON and XML are common formats for data exchange."
    ],
    "database": [
        "A database stores structured data for easy retrieval and management.",
        "SQL is used for relational databases, while NoSQL is used for document-based storage.",
        "You can query, insert, update, and delete data in databases.",
        "Databases are crucial for web and software applications."
    ],
    "devops": [
        "Git is a version control system to track code changes.",
        "GitHub is a platform to host Git repositories and collaborate on code.",
        "Version control helps teams work together and manage code safely."
    ],
    "cloud": [
        "Cloud computing provides computing resources over the internet.",
        "It allows you to host apps, store data, and scale services easily.",
        "Popular cloud providers include AWS, Google Cloud, and Azure."
    ],
    "ai": [
        "AI stands for Artificial Intelligence, where machines simulate human intelligence.",
        "Machine Learning is a subset of AI that lets computers learn from data.",
        "Deep Learning uses neural networks to analyze complex patterns.",
        "Natural Language Processing (NLP) helps computers understand human language.",
        "Computer vision is AI that analyzes images and videos.",
        "Supervised, unsupervised, and reinforcement learning are key AI techniques."
    ],
      "addition": [
        "Just add the numbers together.",
        "Add the two numbers to get the answer."
    ],
    "subtraction": [
        "Subtract the second number from the first.",
        "Find the difference between the two numbers."
    ],
    "multiplication": [
        "Multiply the numbers to get the product.",
        "The answer is the product of the numbers."
    ],
    "division": [
        "Divide the first number by the second.",
        "The answer is the quotient of the numbers."
    ],
    "linear_equation": [
        "Solve the equation to find the value of x.",
        "Isolate x and calculate its value."
    ],
    "area_rectangle": [
        "Multiply length by width to get the area of a rectangle.",
        "Area = length × width."
    ],
    "perimeter_square": [
        "Multiply the side length by 4 to get the perimeter.",
        "Perimeter = 4 × side."
    ],
    "simplify_fraction": [
        "Divide numerator and denominator by their greatest common divisor.",
        "Reduce the fraction to its simplest form."
    ],
    "square_number": [
        "Multiply the number by itself to get the square.",
        "The square of a number is the number times itself."
    ],
    "cube_number": [
        "Multiply the number by itself twice to get the cube.",
        "The cube of a number is number × number × number."
    ],
    "percentage": [
        "Multiply the number by the percentage (as a decimal) to find the result.",
        "Percentage calculation: value × (percentage ÷ 100)."
    ],
    "decimal_to_fraction": [
        "Convert the decimal to a fraction and simplify.",
        "Write the decimal as numerator/denominator in simplest form."
    ],
    "division_equation": [
        "Multiply both sides to isolate x and solve.",
        "Solve for x by reversing the division operation."
    ],
    "vowel": [
        "A vowel is a speech sound produced without blocking the airflow. The main English vowels are a, e, i, o, u.",
        "The five primary vowels in English are A, E, I, O, and U. Sometimes Y can function as a vowel.",
        "Vowel sounds can be short (as in 'cat') or long (as in 'cake').",
        "In the word provided, identify letters a, e, i, o, u as vowels.",
        "Vowels are pronounced with an open mouth and no friction.",
        "Examples of vowel sounds include /a/ as in 'cat', /e/ as in 'bed', /i/ as in 'sit', /o/ as in 'hot', /u/ as in 'sun'.",
        "The vowel in that word can be found among a, e, i, o, u depending on pronunciation.",
        "English vowels form the core sound of syllables in words.",
        "Some words contain multiple vowels that create diphthongs like 'ea' or 'oo'.",
        "Short vowel example: 'pen'. Long vowel example: 'phone'."
    ],

    "consonant": [
        "A consonant is any letter that is not a vowel.",
        "Consonants are speech sounds made with some blockage of airflow.",
        "Examples of consonants include b, c, d, f, g, h, j, k, l, m.",
        "Consonants usually require the tongue, lips, or teeth to restrict airflow.",
        "In English, there are 21 consonant letters.",
        "A consonant sound can be voiced like 'b' or voiceless like 'p'.",
        "Consonants combine with vowels to form syllables.",
        "Letters other than a, e, i, o, u are consonants."
    ],

    "simile": [
        "A simile compares two things using the words 'like' or 'as'.",
        "Similes help create vivid imagery by comparing one thing to another.",
        "Example: 'as brave as a lion' is a simile.",
        "Example: 'She shines like the sun' is a simile.",
        "A simile always contains comparison words such as 'like' or 'as'.",
        "Similes make descriptions more expressive and interesting.",
        "If a sentence compares using 'like' or 'as', it is a simile.",
        "Example: 'as light as a feather' shows comparison clearly.",
        "Writers use similes to make writing more colorful.",
        "To create a simile, compare something to a familiar object."
    ],

    "grammar": [
        "Grammar refers to the rules that govern sentence structure.",
        "A sentence typically has a subject and a predicate.",
        "A noun names a person, place, or thing.",
        "A verb expresses action or state of being.",
        "An adjective describes a noun.",
        "An adverb describes a verb, adjective, or another adverb.",
        "A preposition shows relationship in space or time.",
        "A conjunction joins words or clauses.",
        "An interjection expresses sudden emotion.",
        "Correct grammar ensures clarity and proper communication.",
        "Tenses show the time of an action.",
        "Plural forms often add 's' but some are irregular.",
        "Subject-verb agreement must be maintained in sentences.",
        "A paragraph contains related sentences about one idea."
    ],

    "jamb": [
        "For JAMB-style questions, ensure correct subject-verb agreement.",
        "Check whether the noun is singular or plural.",
        "Irregular plurals include child → children, goose → geese, man → men.",
        "In tense questions, look for time indicators like yesterday or tomorrow.",
        "The subject performs the action in a sentence.",
        "The object receives the action.",
        "Choose options that follow proper grammar rules.",
        "Avoid common errors like 'He go' instead of 'He goes'.",
        "In English exams, pay attention to concord rules.",
        "Vocabulary questions test synonyms and antonyms.",
        "Sentence correction requires identifying grammatical mistakes.",
        "Comprehension questions require understanding context.",
        "Lexis and structure are important in JAMB English.",
        "Always check for agreement between subject and verb."
    ]
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

    # Check greetings first
    if any(word in user_query for word in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]):
        response = random.choice(label_to_answer["greeting"])

    # Check if it's a math question
    elif "solve" in user_query:
        response = solve_math_safe(user_query)

    # Check status queries
    elif any(word in user_query for word in ["how are you", "how’s it going", "what’s up", "how are things"]):
        response = random.choice(label_to_answer["status"])

    else:
        # Use your trained model
        predicted_label = model.predict([user_query])[0]
        responses = label_to_answer.get(predicted_label)
        if responses:
            response = random.choice(responses) if isinstance(responses, list) else responses
        else:
            response = "Sorry, I don't understand that yet."

    return {"answer": response}