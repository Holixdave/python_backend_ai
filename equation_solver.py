# equation_solver.py
import re
from sympy import symbols, Eq, solve, sympify, expand


def preprocess_equation(q: str) -> str:
    q = q.lower()
    q = q.replace("plus", "+")
    q = q.replace("minus", "-")
    q = q.replace("times", "*")
    q = q.replace("multiplied by", "*")
    q = q.replace("divide by", "/")
    q = q.replace("divided by", "/")
    q = q.replace("dived by", "/")
    q = q.replace("equals", "=")
    q = q.replace("equal to", "=")
    q = q.replace("^", "**")
    q = q.replace(" ", "")
    q = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", q)
    q = re.sub(r"([a-zA-Z])(\d)", r"\1*\2", q)
    return q


def solve_equation_with_steps(question: str):
    try:
        q = preprocess_equation(question)

        if "=" not in q:
            return None

        left_str, right_str = q.split("=", 1)

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
            return "No solution found for this equation."

        steps = []
        steps.append("Step 1: Original Equation")
        steps.append(f"   {left_str} = {right_str}")
        steps.append("")
        steps.append("Step 2: Move all terms to one side")
        steps.append(f"   {standard_form} = 0")
        steps.append("")

        if "**2" in str(standard_form):
            steps.append("Step 3: Quadratic Equation — applying quadratic formula")
        else:
            steps.append("Step 3: Solving for the variable")
        steps.append("")

        if isinstance(solution, dict):
            for var, val in solution.items():
                steps.append(f"   {var} = {val}")
        else:
            steps.append(f"   Solution: {solution}")

        steps.append("")
        steps.append("Final Answer:")
        steps.append(f"   {solution}")

        return "\n".join(steps)

    except Exception:
        return "Could not parse or solve this equation. Please write it clearly, e.g.: 2x + 3 = 7"


def solve_multi_variable_equation(equations: list):
    try:
        processed_eqs = [preprocess_equation(eq) for eq in equations]
        all_vars = sorted(set(re.findall(r"[a-z]", " ".join(processed_eqs))))
        if not all_vars:
            return "No variables detected."

        sympy_vars = symbols(all_vars)
        sympy_eqs = []
        for eq in processed_eqs:
            if "=" not in eq:
                return f"Invalid equation format: {eq}"
            left_str, right_str = eq.split("=", 1)
            sympy_eqs.append(Eq(sympify(left_str), sympify(right_str)))

        solutions = solve(sympy_eqs, sympy_vars, dict=True)
        if not solutions:
            return "No solution found."

        steps = ["Solving system of equations:"]
        for i, eq in enumerate(processed_eqs, 1):
            steps.append(f"   Equation {i}: {eq}")
        steps.append("")
        steps.append("Solution:")
        steps.append(f"   {solutions}")
        return "\n".join(steps)

    except Exception as e:
        return f"Could not solve the system: {e}"


if __name__ == "__main__":
    while True:
        user_input = input("Solve equation (or type 'exit'): ").strip()
        if user_input.lower() == "exit":
            break
        if ";" in user_input:
            eq_list = [eq.strip() for eq in user_input.split(";")]
            result = solve_multi_variable_equation(eq_list)
        else:
            result = solve_equation_with_steps(user_input)
        print(result or "I cannot solve that equation.")