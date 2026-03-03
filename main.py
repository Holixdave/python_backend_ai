from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import random
from typing import List
from sympy import symbols, Eq, solve, sympify

app = FastAPI()

# Load trained model
qa_model = joblib.load("ai_model.pkl")
question_gen_model = joblib.load("question_gen_model.pkl")

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
            return "Equation must contain '=', so pls write down the correct structure of the equation"
    
    return "Not a solvable math question"
   
# --------------------------
# Emoji adder
# --------------------------
def add_emoji(response: str, label: str) -> str:
    emoji_map = {
        "greeting": ["👋", "😊"],
        "status": ["🙂", "😄"],
        "student_statement": ["💡", "👍"],
        "math": ["🧮", "✅"],
        "correction": ["❌", "✏️"],
        "question": ["🤔", "🔍"],
        "default": ["✨"]
    }
    emoji = random.choice(emoji_map.get(label, emoji_map["default"]))
    return f"{response} {emoji}"
# Map labels to actual answers
label_to_answer = {
    "greeting": [
        "Hi there! How can I help you today?","Hi there! 👋"
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
         "I’m just a program, but I’m running smoothly!",
        "All systems operational! How can I help?",
        "I’m here and ready to answer your questions!",
        "Everything is working fine on my side!"
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
    ],
    "geography": [
        "Geography is the study of places, the environment, and how humans interact with it.",
        "It involves learning about countries, cities, rivers, and mountains.",
        "The capital of Nigeria is Abuja, a planned city.",
        "Africa's largest river is the Nile, flowing through multiple countries.",
        "Geography helps us understand climate patterns and ecosystems.",
        "Maps and globes are key tools in geography.",
        "It also studies population, cultures, and urban development.",
        "Geography can be physical (landforms, water bodies) or human (society, economy).",
        "Understanding geography helps in navigation and resource management.",
        "Geography is taught to help students learn about the world around them.",
        "It includes natural features like deserts, rivers, and mountains.",
        "Geography explains regional differences and human settlement patterns."
        "Geography studies the Earth's landscapes, environments, and human interactions."
    ],
    "chemistry": [
        "Chemistry is the study of matter, its properties, and how it changes.",
        "Water has the chemical formula H2O.",
        "Table salt is chemically NaCl, sodium chloride.",
        "Acids and bases are important concepts in chemistry.",
        "Chemical reactions involve the rearrangement of atoms.",
        "Organic chemistry studies carbon-based compounds.",
        "Inorganic chemistry studies non-carbon compounds.",
        "Understanding chemistry helps in medicine, industry, and agriculture.",
        "Atoms are the basic building blocks of matter.",
        "Elements and compounds are classified in the periodic table.",
        "Chemistry explains how substances interact at a molecular level.",
        "Stoichiometry allows chemists to calculate reactants and products."
        "Chemistry studies the composition, properties, and reactions of substances."
    ],
    "biology": [
        "Biology is the study of living organisms and their functions.",
        "Photosynthesis is the process by which plants make food from sunlight.",
        "The heart pumps blood throughout the body.",
        "The lungs are responsible for gas exchange.",
        "The kidneys filter waste from the blood.",
        "Reflex actions are automatic responses to stimuli.",
        "The liver has multiple functions, including detoxification.",
        "Penicillin was discovered by Alexander Fleming.",
        "The human body has multiple organ systems working together.",
        "Genetics studies how traits are inherited.",
        "Evolution explains how species change over time.",
        "Cells are the basic units of life."
        "Biology is the study of living organisms and their interactions with the environment."
    ],
    "physics": [
        "Physics is the study of matter, energy, and the laws of nature.",
        "The speed of light is approximately 3 × 10^8 m/s.",
        "Newton's First Law states that an object in motion stays in motion unless acted on by a force.",
        "Gravity pulls objects toward the center of the Earth.",
        "Kinetic energy is the energy of motion.",
        "Acceleration measures the rate of change of velocity.",
        "Ohm's Law relates voltage, current, and resistance.",
        "Inertia is the resistance of an object to a change in motion.",
        "Physics explains natural phenomena and helps design technology.",
        "Force equals mass times acceleration (F = ma).",
        "Energy can be potential, kinetic, or thermal.",
        "Electricity and magnetism are major topics in physics."
    ],
    "mathematics": [
        "Mathematics is the study of numbers, shapes, and patterns.",
        "The Pythagorean theorem states a² + b² = c² for right-angled triangles.",
        "Multiplication is repeated addition.",
        "Squares and cubes are powers of numbers.",
        "Algebra involves solving for unknowns.",
        "Geometry studies shapes, sizes, and properties of space.",
        "Fractions represent parts of a whole.",
        "Decimals and percentages are different ways to express numbers.",
        "Mathematics is essential in science, finance, and engineering.",
        "Equations show relationships between variables.",
        "Linear equations have variables raised to the first power.",
        "Probability measures the likelihood of an event."
        "An algorithm is a step-by-step procedure to solve a problem.",
        "Recursion is when a function calls itself to solve smaller parts of a problem.",
        "Algorithms are essential for computer science and programming.",
        "Efficient algorithms help programs run faster and use less memory."
        "Just add the numbers together.",
        "Add the two numbers to get the answer."
    ],
    "linear_equation": [
        "A linear equation is an equation of the first degree.",
        "It has variables with the highest power of 1.",
        "Example: x + 5 = 12.",
        "Solving involves isolating the variable on one side.",
        "Linear equations have a straight-line graph.",
        "They are used in real-life calculations like finance and measurement.",
        "Two-variable linear equations can be solved simultaneously.",
        "They are foundational in algebra.",
        "The solution is a value that satisfies the equation.",
        "Balancing both sides is key to solving linear equations."
    ],
    "geometry": [
        "Geometry is the study of shapes, sizes, and space.",
        "The area of a rectangle is length × width.",
        "The perimeter of a square is 4 × side.",
        "Triangles can be classified by sides or angles.",
        "Circles have radius, diameter, and circumference.",
        "Angles are measured in degrees.",
        "Pythagoras' theorem applies to right-angled triangles.",
        "Geometry is used in architecture and design.",
        "3D geometry involves cubes, spheres, and cylinders.",
        "Coordinate geometry places shapes on a graph.",
        "Geometric transformations include rotations and reflections."
    ],
    "english_grammar": [
        "A noun names a person, place, or thing.",
        "A verb shows action or state.",
        "An adjective describes a noun.",
        "A pronoun replaces a noun.",
        "A preposition shows relationship between words.",
        "Antonyms are words with opposite meanings.",
        "Synonyms are words with similar meanings.",
        "Vowels are a, e, i, o, u.",
        "Homonyms are words that sound alike but have different meanings.",
        "Grammar rules ensure clarity in writing and speaking.",
        "Reflexive pronouns refer back to the subject.",
        "Conjunctions connect words or clauses."
    ],
    "english_literature": [
        "A simile compares two things using 'like' or 'as'.",
        "A metaphor compares two things without using 'like' or 'as'.",
        "Alliteration repeats the same sound at the start of words.",
        "Onomatopoeia uses words that imitate sounds.",
        "Hyperbole is deliberate exaggeration.",
        "Shakespeare wrote plays like Hamlet and Othello.",
        "Literature reflects culture and society.",
        "A poem often uses rhyme and meter.",
        "Stories can have a moral or lesson.",
        "Character and plot are essential elements of a story.",
        "Literature helps us understand human emotions.",
        "Fables use animals to convey lessons."
    ],
    "arithmetic": [
        "Addition combines numbers.",
        "Subtraction finds the difference between numbers.",
        "Multiplication is repeated addition.",
        "Division splits a number into equal parts.",
        "Order of operations matters: PEMDAS.",
        "Arithmetic is used in everyday calculations.",
        "Even and odd numbers are part of arithmetic.",
        "Prime numbers have exactly two factors.",
        "Factors and multiples are key concepts.",
        "Arithmetic forms the foundation for advanced math.",
        "Decimals and fractions are used in arithmetic operations.",
        "Percentages represent parts of 100."
    ],
    "fractions": [
        "Fractions represent parts of a whole.",
        "Simplifying fractions makes them easier to use.",
        "Equivalent fractions have the same value.",
        "Adding fractions requires a common denominator.",
        "Multiplying fractions multiplies the numerators and denominators.",
        "Dividing fractions involves multiplying by the reciprocal.",
        "Fractions can be converted to decimals and percentages.",
        "Proper fractions are less than 1, improper are equal or more than 1.",
        "Mixed numbers combine a whole number and a fraction.",
        "Fractions are used in recipes, measurements, and finances."
    ],
    "division_equation": [
        "A division equation involves dividing to find a value.",
        "Example: x / 3 = 5, solution: x = 15.",
        "It can be solved by multiplying both sides by the divisor.",
        "Division equations are used in everyday problems.",
        "Check the solution by substituting back into the equation.",
        "They often appear in algebra and word problems.",
        "Understanding division is key to fractions and ratios.",
        "Divide carefully to maintain equality.",
        "Division equations can involve decimals or integers.",
        "Always isolate the variable first for clarity."
    ],
    "percentage": [
        "Percentages express a part of 100.",
        "10% of 50 is 5.",
        "Percentage is used in discounts, interest, and statistics.",
        "Convert percentage to decimal by dividing by 100.",
        "Increase or decrease percentages are used in finance.",
        "Percentage helps compare quantities easily.",
        "A 25% discount on 200 is 50.",
        "Percentages can also be fractions or ratios.",
        "Percentage change = (difference / original) × 100.",
        "Understanding percentages is essential in exams and everyday life."
    ],
     "biology": [
        "Biology is the study of living organisms and their interactions with the environment.",
        "The heart pumps blood throughout the body.",
        "DNA carries genetic information.",
        "Cells are the basic units of life.",
        "Photosynthesis is the process by which plants make food from sunlight.",
        "Osmosis is the movement of water across a semipermeable membrane.",
        "Respiration is how organisms obtain energy from food.",
        "Mitosis is cell division producing identical cells.",
        "Genes determine inherited traits.",
        "The largest human organ is the skin.",
        "The function of the lungs is to exchange oxygen and carbon dioxide.",
        "Enzymes speed up chemical reactions in the body.",
        "Hormones regulate body processes.",
        "Vaccines help prevent diseases.",
        "Chlorophyll absorbs sunlight for photosynthesis."
    ],

    "chemistry": [
        "An acid is a substance with a pH less than 7.",
        "A base is a substance with a pH greater than 7.",
        "Neutralization occurs when an acid reacts with a base.",
        "Atoms are the basic building blocks of matter.",
        "Molecules are made of two or more atoms bonded together.",
        "Elements are pure substances consisting of one type of atom.",
        "Metals are good conductors of heat and electricity.",
        "A chemical reaction transforms substances into new products.",
        "Noble gases are inert and do not easily react.",
        "Water is formed from hydrogen and oxygen.",
        "The periodic table organizes elements by their properties.",
        "Oxidation involves the loss of electrons.",
        "Reduction involves the gain of electrons.",
        "Electrolysis splits compounds using electricity.",
        "Catalysts speed up reactions without being consumed."
    ],

    "physics": [
        "Force is a push or pull on an object.",
        "Newton's First Law states an object remains at rest or in motion unless acted upon.",
        "Kinetic energy is energy of motion.",
        "Potential energy is stored energy.",
        "Gravity pulls objects toward the earth.",
        "Acceleration is the rate of change of velocity.",
        "Mass is the amount of matter in an object.",
        "Friction opposes motion.",
        "Inertia is an object's resistance to change in motion.",
        "A wave transfers energy without transferring matter.",
        "Sound is a mechanical wave.",
        "Light is an electromagnetic wave.",
        "Reflection occurs when light bounces off a surface.",
        "Refraction is the bending of light through different mediums.",
        "Electricity is the flow of electric charge."
    ],

    "geometry": [
        "A triangle has three sides.",
        "A square has four equal sides.",
        "A rectangle has opposite sides equal.",
        "A circle is a set of points equidistant from a center.",
        "The radius is the distance from the center to the edge.",
        "The diameter is twice the radius.",
        "The perimeter is the total length around a shape.",
        "The area is the space inside a shape.",
        "The volume measures the space occupied by an object.",
        "Angles are formed where two lines meet.",
        "An equilateral triangle has all sides equal.",
        "A right angle measures 90 degrees.",
        "A parallelogram has opposite sides parallel.",
        "A trapezium has only one pair of parallel sides.",
        "The Pythagorean theorem relates the sides of a right triangle."
    ],

    "arithmetic": [
        "Addition combines numbers together.",
        "Subtraction finds the difference between numbers.",
        "Multiplication is repeated addition.",
        "Division splits a number into equal parts.",
        "50% of 200 is 100.",
        "The sum of 12 and 7 is 19.",
        "Subtracting 5 from 20 gives 15.",
        "Multiplying 8 by 6 gives 48.",
        "Dividing 45 by 9 gives 5.",
        "To calculate percentages, divide part by whole and multiply by 100.",
        "Even numbers are divisible by 2.",
        "Odd numbers leave a remainder of 1 when divided by 2.",
        "Prime numbers are greater than 1 and divisible only by 1 and itself.",
        "Composite numbers have more than two factors.",
        "The order of operations is parentheses, exponents, multiplication/division, addition/subtraction."
    ],

    "linear_equation": [
        "To solve x + 5 = 12, subtract 5 from both sides to get x = 7.",
        "For 3x = 21, divide both sides by 3 to get x = 7.",
        "To solve 2x - 4 = 10, add 4 then divide by 2 to get x = 7.",
        "Linear equations have variables with power 1.",
        "You can isolate the variable using inverse operations.",
        "Check solutions by substituting back into the original equation.",
        "Balance both sides of the equation.",
        "Simplify terms before solving.",
        "Equations can have one solution, no solution, or infinite solutions.",
        "Graphing a linear equation gives a straight line.",
        "Slope and intercept help describe a line.",
        "Parallel lines have the same slope.",
        "Perpendicular lines have slopes that are negative reciprocals.",
        "Use substitution or elimination for systems of linear equations.",
        "Coefficients are the numbers multiplying variables."
    ],

    "square_number": [
        "The square of 7 is 49.",
        "To square a number, multiply it by itself.",
        "4² = 16.",
        "5² = 25.",
        "10² = 100.",
        "12² = 144.",
        "Squaring increases the number quickly.",
        "A perfect square has an integer as its square root.",
        "The sum of consecutive odd numbers forms perfect squares.",
        "Negative numbers squared become positive.",
        "Square numbers appear on the diagonal of Pascal’s Triangle.",
        "The square of zero is zero.",
        "Squares are used in calculating areas of squares.",
        "The square of 15 is 225.",
        "The square of 20 is 400."
    ],

    "cube_number": [
        "The cube of 3 is 27.",
        "To cube a number, multiply it by itself twice.",
        "4³ = 64.",
        "5³ = 125.",
        "2³ = 8.",
        "10³ = 1000.",
        "Cubes grow faster than squares.",
        "The cube of 0 is 0.",
        "Negative numbers cubed remain negative.",
        "Cube numbers are used in volume calculations.",
        "The cube of 6 is 216.",
        "The cube of 7 is 343.",
        "The cube of 8 is 512.",
        "The cube of 9 is 729.",
        "The cube of 12 is 1728."
    ],

    "english_literature": [
        "A simile compares two things using 'like' or 'as'.",
        "Example: 'Her smile was like sunshine.'",
        "A metaphor compares two things without using 'like' or 'as'.",
        "Example: 'Time is a thief.'",
        "Alliteration is the repetition of initial consonant sounds.",
        "Example: 'She sells sea shells.'",
        "Personification gives human traits to non-human things.",
        "Example: 'The wind whispered.'",
        "Hyperbole is an exaggeration.",
        "Example: 'I'm so hungry I could eat a horse.'",
        "Onomatopoeia imitates sounds.",
        "Example: 'The bees buzzed.'",
        "Irony expresses the opposite of the literal meaning.",
        "Example: 'A fire station burns down.'",
        "Symbolism uses symbols to represent ideas."
    ],

    "english_grammar": [
        "A noun is a person, place, thing, or idea.",
        "A verb expresses an action or state.",
        "An adjective describes a noun.",
        "An adverb modifies a verb, adjective, or adverb.",
        "A pronoun replaces a noun.",
        "A preposition shows relationship between nouns.",
        "Conjunctions connect words or phrases.",
        "Interjections express strong emotion.",
        "Articles are 'a', 'an', 'the'.",
        "Singular refers to one, plural refers to many.",
        "Subjects perform the action of the verb.",
        "Objects receive the action of the verb.",
        "Tenses indicate time of action.",
        "Active voice shows the subject performs the action.",
        "Passive voice shows the subject receives the action."
    ],

    "geography": [
        "Nigeria's capital is Abuja.",
        "The Nile is the longest river in the world.",
        "A continent is a large landmass.",
        "Climate describes long-term weather patterns.",
        "An ecosystem is a community of living organisms.",
        "The Earth has layers: crust, mantle, core.",
        "Longitude and latitude locate positions on Earth.",
        "The Sahara is the largest desert.",
        "The Equator divides the Earth into north and south.",
        "Mount Everest is the highest mountain.",
        "Oceans cover about 71% of Earth's surface.",
        "Rainforests are dense, tropical forests.",
        "Deserts have low rainfall.",
        "Islands are land surrounded by water.",
        "Peninsulas are land surrounded by water on three sides."
    ],

    "technology": [
        "A computer is an electronic device for processing data.",
        "AI stands for Artificial Intelligence.",
        "An algorithm is a set of instructions to solve a problem.",
        "Machine learning allows computers to learn from data.",
        "Blockchain is a digital ledger.",
        "The Internet connects computers worldwide.",
        "Databases store organized information.",
        "HTML is a markup language for web pages.",
        "CSS styles web content.",
        "JavaScript makes websites interactive.",
        "Networks allow devices to communicate.",
        "Software is a set of instructions for computers.",
        "Hardware refers to physical components.",
        "Cloud computing stores data online.",
        "Servers provide resources or services to clients."
    ],

    "history": [
        "William Shakespeare was an English playwright.",
        "The Renaissance was a period of cultural revival.",
        "Ancient Egypt and Mesopotamia were early civilizations.",
        "The Industrial Revolution transformed economies.",
        "Christopher Columbus discovered America in 1492.",
        "Nelson Mandela fought against apartheid.",
        "Democracy allows people to vote.",
        "Monarchy is rule by a king or queen.",
        "Slavery involved forced labor of humans.",
        "World War I occurred from 1914 to 1918.",
        "World War II occurred from 1939 to 1945.",
        "The Berlin Wall fell in 1989.",
        "The Magna Carta limited the power of kings.",
        "The French Revolution occurred in 1789.",
        "The Cold War was a geopolitical tension after WWII."
    ],

    "economics": [
        "Supply and demand affect prices.",
        "Inflation is the rise of prices over time.",
        "GDP measures a country's economic output.",
        "A recession is a period of economic decline.",
        "Trade balance is the difference between exports and imports.",
        "Investment is putting money into assets for returns.",
        "Entrepreneurship involves starting businesses.",
        "Taxation is collecting money by governments.",
        "Fiscal policy controls spending and taxation.",
        "Market economy relies on supply and demand.",
        "Deflation occurs when prices fall.",
        "Opportunity cost is the value of what you give up.",
        "Monetary policy controls the money supply.",
        "Capitalism encourages private ownership.",
        "Socialism promotes government control of resources."
    ],
    "astronomy": [
        "The Solar System has 8 planets.",
        "A black hole has gravity so strong that nothing escapes.",
        "A comet is made of ice and dust.",
        "Stars are luminous balls of gas.",
        "A galaxy is a collection of stars and systems.",
        "The Milky Way is our galaxy.",
        "The Moon orbits the Earth.",
        "Eclipses occur when one body blocks another.",
        "Satellites orbit planets.",
        "Phases of the Moon include new, crescent, quarter, gibbous, and full.",
        "Light-years measure distance in space.",
        "Asteroids are rocky objects in space.",
        "The Sun is a star at the center of our Solar System.",
        "Planets revolve around stars.",
        "Gravity keeps planets in orbit."
    ],

    "percentage": [
        "To calculate 50%, multiply the number by 0.5.",
        "25% of 80 is 20.",
        "10% of 200 is 20.",
        "Percentage represents parts per hundred.",
        "Increase by 20% means multiply by 1.2.",
        "Decrease by 15% means multiply by 0.85.",
        "50% off a price halves it.",
        "75% means three-quarters.",
        "Percentages are used in finance and exams.",
        "To convert a fraction to percentage, divide numerator by denominator and multiply by 100.",
        "100% is the whole amount.",
        "Percentage change = (new - old)/old × 100.",
        "The total of percentages in a pie chart is 100%.",
        "Percentage error = |measured - actual| / actual × 100.",
        "Increase and decrease percentages are common in statistics."
    ],

    "fraction": [
        "0.25 as a fraction is 1/4.",
        "0.2 as a fraction is 1/5.",
        "0.125 as a fraction is 1/8.",
        "Simplify 15/20 to 3/4.",
        "3/6 simplified is 1/2.",
        "Add fractions with common denominators.",
        "Subtract fractions with common denominators.",
        "Multiply fractions across numerators and denominators.",
        "Divide fractions by flipping the second fraction and multiplying.",
        "Improper fractions have numerator greater than denominator.",
        "Mixed numbers have whole and fractional parts.",
        "Reciprocal of a fraction flips numerator and denominator.",
        "Fractions represent parts of a whole.",
        "Decimal to fraction conversion involves denominator as a power of 10.",
        "Lowest common denominator is used for adding fractions."
    ],
    "biology": [
        "Biology is the study of living organisms and their life processes.",
        "It focuses on cells, genetics, ecology, and evolution.",
        "The human body has organs like the heart and lungs that biology studies.",
        "Photosynthesis in plants is a key topic in biology.",
        "DNA carries genetic information in living beings.",
        "Osmosis is the movement of water across membranes.",
        "Mitosis is the process of cell division.",
        "Hormones regulate body functions and are studied in biology.",
        "Respiration is how cells generate energy from food.",
        "Enzymes speed up chemical reactions in the body.",
        "Vaccines help protect against diseases by triggering immunity.",
        "Chlorophyll gives plants their green color and helps in photosynthesis.",
        "Genes are units of heredity that determine traits.",
        "The largest human organ is the skin.",
        "Biology examines how organisms adapt to their environment."
    ],

    "chemistry": [
        "Chemistry is the study of matter and its interactions.",
        "An acid releases hydrogen ions in a solution.",
        "A base releases hydroxide ions in a solution.",
        "Neutralization is when an acid and base react to form water and salt.",
        "Atoms are the building blocks of matter.",
        "Molecules are combinations of atoms.",
        "Elements are pure substances consisting of one type of atom.",
        "Chemical reactions transform substances into new ones.",
        "Noble gases are inert and rarely react with other elements.",
        "Water is formed from hydrogen and oxygen.",
        "The periodic table organizes elements by properties.",
        "Oxidation involves the loss of electrons.",
        "Reduction involves the gain of electrons.",
        "Electrolysis uses electricity to break compounds.",
        "Catalysts speed up reactions without being consumed."
    ],

    "physics": [
        "Physics studies matter, energy, and their interactions.",
        "Force is a push or pull on an object.",
        "Newton's First Law states an object in motion stays in motion unless acted upon.",
        "Kinetic energy is energy of motion.",
        "Potential energy is stored energy.",
        "Gravity pulls objects toward Earth.",
        "Acceleration is the rate of change of velocity.",
        "Mass measures how much matter an object contains.",
        "Friction resists motion between surfaces.",
        "Inertia is an object's resistance to change in motion.",
        "Waves transfer energy without moving matter.",
        "Sound is a vibration that travels through a medium.",
        "Light is an electromagnetic wave.",
        "Reflection occurs when light bounces off a surface.",
        "Refraction is the bending of light as it passes through different media."
    ],

    "geometry": [
        "A triangle has three sides.",
        "A square has four equal sides.",
        "A rectangle has opposite sides equal.",
        "A circle is a set of points equidistant from a center.",
        "Radius is the distance from center to edge of a circle.",
        "Diameter is twice the radius.",
        "Perimeter is the total length around a shape.",
        "Area is the space inside a shape.",
        "Volume is the space a 3D object occupies.",
        "Angles are measured in degrees.",
        "Equilateral triangles have all sides equal.",
        "A right angle measures 90 degrees.",
        "A parallelogram has opposite sides parallel.",
        "A trapezium has only one pair of parallel sides.",
        "The Pythagorean theorem relates sides of a right triangle."
    ],

    "arithmetic": [
        "Addition combines numbers to get a sum.",
        "Subtraction finds the difference between numbers.",
        "Multiplication finds the product of numbers.",
        "Division splits numbers into equal parts.",
        "50% of 200 is 100.",
        "Even numbers are divisible by 2.",
        "Odd numbers leave a remainder of 1 when divided by 2.",
        "Prime numbers have exactly two factors.",
        "Composite numbers have more than two factors.",
        "Percentages are a way to express fractions out of 100.",
        "Order of operations: parentheses, exponents, multiplication/division, addition/subtraction.",
        "Multiplying by zero gives zero.",
        "Subtracting zero keeps the number unchanged.",
        "Adding zero keeps the number unchanged.",
        "Dividing by one keeps the number unchanged."
    ],

    "linear_equation": [
        "Linear equations have variables raised only to the first power.",
        "To solve x + 5 = 12, subtract 5 from both sides.",
        "3x = 21 is solved by dividing both sides by 3.",
        "2x - 4 = 10 is solved by adding 4 then dividing by 2.",
        "Graphing linear equations shows a straight line.",
        "Slope measures steepness of a line.",
        "Y-intercept is where the line crosses the y-axis.",
        "Substitution method can solve linear systems.",
        "Check solutions by plugging them back into the equation.",
        "Balance both sides of the equation to maintain equality.",
        "Coefficients multiply the variables.",
        "Parallel lines have the same slope.",
        "Perpendicular lines have slopes that are negative reciprocals.",
        "Simplify terms first before solving.",
        "Linear equations model real-world problems like speed or distance."
    ],

    "english": [
        "A vowel is a speech sound like a, e, i, o, u.",
        "A consonant is any letter that is not a vowel.",
        "A simile compares two things using 'like' or 'as'.",
        "Example: Her smile was like sunshine.",
        "A metaphor compares without using 'like' or 'as'.",
        "Nouns are names of people, places, or things.",
        "Verbs describe actions or states.",
        "Adjectives describe nouns.",
        "Adverbs describe verbs, adjectives, or other adverbs.",
        "Prepositions show relationship between nouns/pronouns and other words.",
        "Interjections express strong emotions.",
        "Pronouns replace nouns to avoid repetition.",
        "Adjectives vs adverbs: adjectives describe nouns, adverbs describe verbs/adjectives.",
        "Identify similes in poetry or prose.",
        "Correct grammar improves clarity in writing."
    ],

    "reasoning": [
        "Logic is the study of correct reasoning.",
        "An argument is a set of statements supporting a conclusion.",
        "Deduction derives specific conclusions from general rules.",
        "Induction infers general rules from specific examples.",
        "A fallacy is an error in reasoning.",
        "A premise is a supporting statement in an argument.",
        "A conclusion is the final statement derived from premises.",
        "Example of deductive reasoning: All men are mortal; Socrates is a man; Socrates is mortal.",
        "Example of inductive reasoning: Observing swans are white, inferring all swans are white.",
        "Cause and effect relationships are key in reasoning.",
        "Analogy compares two similar situations.",
        "Critical thinking helps identify assumptions.",
        "Logical errors weaken arguments.",
        "Reasoning improves problem-solving skills.",
        "Reasoning is used in JAMB questions to analyze patterns."
    ],

    "statistics": [
        "Probability measures likelihood of events.",
        "Mean is the average of numbers.",
        "Median is the middle value of numbers.",
        "Mode is the most frequent value.",
        "Standard deviation measures data spread.",
        "Variance measures how far numbers are from the mean.",
        "Probability of tossing a coin: 50% heads, 50% tails.",
        "Probability of rolling a die: 1/6 for each number.",
        "Probability in cards depends on deck composition.",
        "Statistics help interpret real-world data.",
        "Sample space lists all possible outcomes.",
        "Expected value predicts average outcome over time.",
        "Random events are unpredictable.",
        "Use statistics to solve JAMB math and reasoning problems.",
        "Mean, median, mode are measures of central tendency."
    ],

    "ai": [
        "Artificial Intelligence enables machines to mimic human intelligence.",
        "Machine Learning is a subset of AI that learns from data.",
        "Deep Learning uses neural networks for complex tasks.",
        "NLP is Natural Language Processing, understanding human language.",
        "Computer vision lets machines 'see' and interpret images.",
        "Supervised learning uses labeled data for training.",
        "Unsupervised learning finds patterns in unlabeled data.",
        "Reinforcement learning learns via rewards and punishments.",
        "AI is used in healthcare to predict diseases.",
        "AI is applied in education for personalized learning.",
        "AI in finance detects fraud and predicts trends.",
        "Ethics in AI ensures responsible usage.",
        "AI has limitations and cannot fully replace humans.",
        "Future AI may impact jobs and society.",
        "Benefits of AI include efficiency, accuracy, and automation."
    ],
}


# Input models
class AIQuery(BaseModel):
    query: str
class GeneratePrompt(BaseModel):
    prompt: str  # e.g., "Generate a Python question"
def solve_math_safe(question):
    x = symbols('x')
    if "solve" in question.lower() and "=" in question:
        lhs, rhs = question.split("=")
        try:
            eq = Eq(sympify(lhs.strip()), sympify(rhs.strip()))
            solution = solve(eq, x)
            return f"Solution: {solution}"
        except Exception as e:
            return f"Error parsing equation: {e}"
    return None

# A simple function to detect conversation or student statement
def detect_conversation(query: str) -> str:
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if any(word in query.lower() for word in greetings):
        return "greeting"
    elif "how are you" in query.lower():
        return "status"
    elif "i think" in query.lower() or "i believe" in query.lower():
        return "student_statement"
    return "question"


@app.post("/ai-query")
async def ai_query(request: AIQuery):
    user_query = request.query.strip()
    mode = detect_conversation(user_query)
    response = ""

    if mode in ["greeting", "status"]:
        response = random.choice(label_to_answer.get(mode, ["Hello! How can I help you?"]))
    elif mode == "student_statement":
        response = f"I see you said: '{user_query}'. "
        # Example correction logic
        if "2+2=5" in user_query:
            response += "Actually, 2 + 2 = 4."
            response = add_emoji(response, "correction")
        else:
            response += "That seems reasonable, keep thinking!"
            response = add_emoji(response, "student_statement")
    else:
        # Try math solver first
        math_solution = solve_math_safe(user_query)
        if math_solution:
            response = add_emoji(math_solution, "math")
        else:
            # Use label prediction (your ML model)
            predicted_label = model.predict([user_query])[0]
            responses = label_to_answer.get(predicted_label)
            if responses:
                if isinstance(responses, list):
                    response = random.choice(responses)
                else:
                    response = responses
            else:
                response = "I'm not sure about that. Can you clarify?"
            response = add_emoji(response, predicted_label if responses else "question")

    return {"answer": response}

# -------------------------
# Endpoint to generate question
# -------------------------
@app.post("/generate-question")
async def generate_question(request: GeneratePrompt):
    user_prompt = request.prompt.strip()
    
    # Predict a completion
    try:
        predicted_questions = question_gen_model.predict([user_prompt])
        # pick a random one if multiple returned (for variation)
        question = random.choice(predicted_questions)
        return {"question": question}
    except Exception as e:
        return {"error": f"Failed to generate question: {e}"}