# math_solver.py
import re

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
    q = q.replace("squared", "**2")
    q = re.sub(r"(\d+) to the power of (\d+)", r"\1**\2", q)
    plus = re.search(r'(\d+)\s*plus\s*(\d+)', q)
    minus = re.search(r'(\d+)\s*(minus|subtract)\s*(\d+)', q)
    multiply = re.search(r'(\d+)\s*(times|multiply|x)\s*(\d+)', q)
    divide = re.search(r'(\d+)\s*(divided by|/)\s*(\d+)', q)
    square = re.search(r'square of (\d+)', q)
    sqrt = re.search(r'square root of (\d+)', q)

    if plus:
        a, b = int(plus[1]), int(plus[2])
        return f"{a} + {b} = {a + b}"
    if minus:
        a, b = int(minus[1]), int(minus[3])
        return f"{a} - {b} = {a - b}"
    if multiply:
        a, b = int(multiply[1]), int(multiply[3])
        return f"{a} × {b} = {a * b}"
    if divide:
        a, b = int(divide[1]), int(divide[3])
        return f"{a} ÷ {b} = {a / b}"
    if square:
        a = int(square[1])
        return f"{a}² = {a ** 2}"
    if sqrt:
        a = int(sqrt[1])
        return f"√{a} = {a ** 0.5}"
     # Handle "subtract X from Y" -> "Y - X"
    subtract_match = re.match(r"subtract (\d+\.?\d*) from (\d+\.?\d*)", q)
    if subtract_match:
        q = f"{subtract_match.group(2)} - {subtract_match.group(1)}"

    # Keep only numbers, operators, parentheses, decimal points
    q_clean = re.sub(r"[^0-9+\-*/%.() ]", "", q)

    try:
        # Evaluate result
        result = eval(q_clean)
        # Create an explanation string with friendly emoji
        explanation = (
            f"🧮 Solving your math: {q_clean} = {result}\n"
            f"✅ The answer is {result}. This is your solving here: {q_clean} = {result}"
        )
        return explanation
    except Exception:
        return None


# Example standalone usage
if __name__ == "__main__":
    while True:
        user_input = input("Math question: ").strip()
        if user_input.lower() == "exit":
            break
        output = solve_math_with_explanation(user_input)
        print(output or "❌ I cannot solve that math question.")