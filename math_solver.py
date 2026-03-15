# math_solver.py
import re
import math


def solve_math_with_explanation(question: str) -> str | None:
    q = question.lower()
    q = re.sub(r"what is|calculate|compute|please|the result of|find|\?", "", q)

    q = q.replace("plus", "+").replace("add", "+")
    q = q.replace("minus", "-").replace("subtract", "-")
    q = q.replace("times", "*").replace("multiplied by", "*").replace("multiply", "*")
    q = q.replace("divided by", "/").replace("divide", "/").replace("over", "/")
    q = q.replace("modulus", "%").replace("mod", "%")
    q = re.sub(r"(\d+)\s*to\s*the\s*power\s*of\s*(\d+)", r"\1**\2", q)

    square = re.search(r'square\s*of\s*(\d+)', q)
    sqrt = re.search(r'square\s*root\s*of\s*(\d+)', q)
    cube = re.search(r'cube\s*of\s*(\d+)', q)
    percent = re.search(r'(\d+\.?\d*)\s*%\s*of\s*(\d+\.?\d*)', q)

    if square:
        a = int(square.group(1))
        return f"{a}² = {a ** 2}"

    if sqrt:
        a = int(sqrt.group(1))
        return f"√{a} = {round(math.sqrt(a), 4)}"

    if cube:
        a = int(cube.group(1))
        return f"{a}³ = {a ** 3}"

    if percent:
        p, total = float(percent.group(1)), float(percent.group(2))
        result = (p / 100) * total
        return f"{p}% of {total} = {result}"

    subtract_match = re.search(r"subtract\s*(\d+\.?\d*)\s*from\s*(\d+\.?\d*)", q)
    if subtract_match:
        q = f"{subtract_match.group(2)} - {subtract_match.group(1)}"

    q_clean = re.sub(r"[^0-9+\-*/%.() \^]", "", q).strip()
    q_clean = q_clean.replace("^", "**")

    if not q_clean:
        return None

    try:
        result = eval(q_clean)
        if isinstance(result, float):
            result = round(result, 4)
        explanation = (
            f"Calculating: {q_clean}\n"
            f"Answer: {result}"
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
        print(output or "I cannot solve that math question.")