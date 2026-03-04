# equation_solver.py
import re
from sympy import symbols, Eq, solve, sympify, expand

def solve_equation_with_steps(question: str):

    q = question.lower()

    # Normalize words
    q = q.replace("plus", "+")
    q = q.replace("minus", "-")
    q = q.replace("times", "*")
    q = q.replace("multiplied by", "*")
    q = q.replace("divide by", "/")
    q = q.replace("divided by", "/")
    q = q.replace("equals", "=")
    q = q.replace("^", "**")
    q = q.replace(" ", "")

    if "=" not in q:
        return None

    try:
        # 🔥 Fix implicit multiplication (2x → 2*x)
        q = re.sub(r"(\d)([a-z])", r"\1*\2", q)

        left_str, right_str = q.split("=")

        vars_found = sorted(set(re.findall(r"[a-z]", q)))
        if not vars_found:
            return None

        sympy_vars = symbols(vars_found)

        left_expr = sympify(left_str)
        right_expr = sympify(right_str)

        equation = Eq(left_expr, right_expr)

        standard_form = expand(left_expr - right_expr)

        solution = solve(equation, sympy_vars)

        if not solution:
            return "❌ No solution found."

        steps = []
        steps.append("📘 Step 1: Original Equation")
        steps.append(f"   {left_str} = {right_str}")
        steps.append("")
        steps.append("📗 Step 2: Move all terms to one side")
        steps.append(f"   {standard_form} = 0")
        steps.append("")

        if "**2" in str(standard_form):
            steps.append("📙 Step 3: This is a Quadratic Equation")
        else:
            steps.append("📙 Step 3: Solve for variable(s)")

        steps.append("")

        if isinstance(solution, dict):
            for var in solution:
                steps.append(f"   {var} = {solution[var]}")
        else:
            steps.append(f"   Solution: {solution}")

        steps.append("")
        steps.append("✅ Final Answer:")
        steps.append(f"   {solution}")

        return "\n".join(steps)

    except Exception:
        return None