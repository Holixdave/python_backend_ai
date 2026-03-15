# math_solver.py
import re
import math

def solve_math_with_explanation(question: str) -> str | None:
    """
    Solve arithmetic questions and return a step-by-step style explanation.
    Returns None if it cannot parse as math.
    """
    q = question.lower()
    q = re.sub(r"what is|calculate|compute|please|the result of|\?", "", q)

    # Replace worded operations with symbols
    q = q.replace("plus", "+").replace("add", "+")
    q = q.replace("minus", "-").replace("subtract", "-")
    q = q.replace("times", "*").replace("multiplied by", "*").replace("multiply", "*")
    q = q.replace("divided by", "/").replace("divide", "/").replace("over", "/")
    q = q.replace("modulus", "%").replace("mod", "%")
    q = re.sub(r"(\d+) to the power of (\d+)", r"\1**\2", q)

    # BUG FIX: These regex checks must happen BEFORE stripping non-numeric chars
    plus = re.search(r'(\d+)\s*\+\s*(\d+)', q)
    minus_m = re.search(r'(\d+)\s*-\s*(\d+)', q)
    multiply = re.search(r'(\d+)\s*\*\s*(\d+)', q)
    divide = re.search(r'(\d+)\s*/\s*(\d+)', q)
    square = re.search(r'square of (\d+)', q)
    sqrt = re.search(r'square root of (\d+)', q)

    if square:
        a = int(square.group(1))
        return f"{a}² = {a ** 2}"   # BUG FIX: was "a  2"

    if sqrt:
        a = int(sqrt.group(1))
        return f"√{a} = {math.sqrt(a)}"  # BUG FIX: was "a  0.5"

    if plus:
        a, b = int(plus.group(1)), int(plus.group(2))
        return f"{a} + {b} = {a + b}"

    if minus_m:
        a, b = int(minus_m.group(1)), int(minus_m.group(2))
        return f"{a} - {b} = {a - b}"

    if multiply:
        a, b = int(multiply.group(1)), int(multiply.group(2))
        return f"{a} × {b} = {a * b}"

    if divide:
        a, b = int(divide.group(1)), int(divide.group(2))
        if b == 0:
            return "❌ Cannot divide by zero."
        return f"{a} ÷ {b} = {a / b}"

    # Handle "subtract X from Y" -> "Y - X"
    subtract_match = re.search(r"subtract (\d+\.?\d*) from (\d+\.?\d*)", q)
    if subtract_match:
        q = f"{subtract_match.group(2)} - {subtract_match.group(1)}"

    # Keep only numbers, operators, parentheses, decimal points
    q_clean = re.sub(r"[^0-9+\-*/%.() ]", "", q).strip()

    if not q_clean:
        return None

    try:
        result = eval(q_clean)
        explanation = (
            f"🧮 Solving: {q_clean} = {result}\n"
            f"✅ The answer is {result}."
        )
        return explanation
    except Exception:
        return None


if __name__ == "__main__":
    while True:
        user_input = input("Math question: ").strip()
        if user_input.lower() == "exit":
            break
        output = solve_math_with_explanation(user_input)
        print(output or "❌ I cannot solve that math question.")