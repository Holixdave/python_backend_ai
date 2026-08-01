# math_solver.py
# ─────────────────────────────────────────────────────────────────────────────
# Enhanced math solver — handles:
#   Basic arithmetic, percentages, powers, roots
#   Algebra (linear, quadratic, simultaneous equations)
#   Geometry (area, perimeter, volume, angles)
#   Statistics (mean, median, mode, range, variance, std dev)
#   Trigonometry (sin, cos, tan and inverses)
#   Logarithms and exponentials
#   Word problems (profit/loss, speed/distance/time, ratio, fraction)
#   Step-by-step explanations for everything
# ─────────────────────────────────────────────────────────────────────────────

import re
import math
from fractions import Fraction
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# NORMALISER — cleans natural language into solvable expressions
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(q: str) -> str:
    q = q.lower().strip()

    # Strip filler words
    fillers = [
        "what is", "what's", "calculate", "compute", "find the value of",
        "find", "evaluate", "solve", "determine", "please", "kindly",
        "the result of", "the answer to", "give me", "tell me",
    ]
    for f in fillers:
        q = q.replace(f, " ")

    # Word → operator
    q = re.sub(r'\bplus\b',          '+',  q)
    q = re.sub(r'\badd\b',           '+',  q)
    q = re.sub(r'\bminus\b',         '-',  q)
    q = re.sub(r'\bsubtract\b',      '-',  q)
    q = re.sub(r'\btimes\b',         '*',  q)
    q = re.sub(r'\bmultiplied by\b', '*',  q)
    q = re.sub(r'\bmultiply\b',      '*',  q)
    q = re.sub(r'\bdivided by\b',    '/',  q)
    q = re.sub(r'\bdivide\b',        '/',  q)
    q = re.sub(r'\bover\b',          '/',  q)
    q = re.sub(r'\bmod\b',           '%',  q)
    q = re.sub(r'\bmodulo\b',        '%',  q)
    q = re.sub(r'\^',                '**', q)
    q = re.sub(r'\bsquared\b',       '**2', q)
    q = re.sub(r'\bcubed\b',         '**3', q)

    # "N to the power of M"
    q = re.sub(
        r'(\d+\.?\d*)\s*to\s*the\s*power\s*of\s*(\d+\.?\d*)',
        r'\1**\2', q
    )

    # "subtract A from B" → "B - A"
    m = re.search(
        r'subtract\s*(\d+\.?\d*)\s*from\s*(\d+\.?\d*)', q)
    if m:
        q = q.replace(m.group(0), f'{m.group(2)} - {m.group(1)}')

    return q.strip()


# ─────────────────────────────────────────────────────────────────────────────
# ARITHMETIC — basic eval with safety
# ─────────────────────────────────────────────────────────────────────────────

def _safe_eval(expr: str) -> Optional[float]:
    """Evaluates a numeric expression safely."""
    expr = re.sub(r'[^0-9+\-*/%.() ]', '', expr).strip()
    if not expr:
        return None
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return round(float(result), 6)
    except Exception:
        return None


def _fmt(n: float) -> str:
    """Format: remove trailing zeros, show fractions for clean values."""
    if n == int(n):
        return str(int(n))
    return f"{n:.6f}".rstrip('0').rstrip('.')


# ─────────────────────────────────────────────────────────────────────────────
# SPECIAL FORMS
# ─────────────────────────────────────────────────────────────────────────────

def _try_special(q_raw: str) -> Optional[str]:
    q = _normalise(q_raw)
    orig = q_raw.lower()

    # ── Percentage ──────────────────────────────────────────────────────────
    m = re.search(r'(\d+\.?\d*)\s*%\s*of\s*(\d+\.?\d*)', q)
    if m:
        p, total = float(m.group(1)), float(m.group(2))
        result = (p / 100) * total
        return (
            f"Percentage Calculation\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Formula  : (percentage ÷ 100) × total\n"
            f"Working  : ({p} ÷ 100) × {total}\n"
            f"         = {p/100} × {total}\n"
            f"Answer   : {_fmt(result)}"
        )

    # ── Square ───────────────────────────────────────────────────────────────
    m = re.search(r'square\s*(?:of\s*)?(\d+\.?\d*)', orig)
    if m:
        a = float(m.group(1))
        return (
            f"Square of {_fmt(a)}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Formula  : n²\n"
            f"Working  : {_fmt(a)} × {_fmt(a)}\n"
            f"Answer   : {_fmt(a**2)}"
        )

    # ── Square root ──────────────────────────────────────────────────────────
    m = re.search(r'square\s*root\s*(?:of\s*)?(\d+\.?\d*)', orig)
    if m:
        a = float(m.group(1))
        result = math.sqrt(a)
        return (
            f"Square Root of {_fmt(a)}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Formula  : √n\n"
            f"Answer   : {_fmt(result)}"
        )

    # ── Cube ─────────────────────────────────────────────────────────────────
    m = re.search(r'cube\s*(?:of\s*)?(\d+\.?\d*)', orig)
    if m:
        a = float(m.group(1))
        return (
            f"Cube of {_fmt(a)}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Formula  : n³\n"
            f"Working  : {_fmt(a)} × {_fmt(a)} × {_fmt(a)}\n"
            f"Answer   : {_fmt(a**3)}"
        )

    # ── Cube root ────────────────────────────────────────────────────────────
    m = re.search(r'cube\s*root\s*(?:of\s*)?(\d+\.?\d*)', orig)
    if m:
        a = float(m.group(1))
        result = a ** (1/3)
        return (
            f"Cube Root of {_fmt(a)}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Formula  : ∛n\n"
            f"Answer   : {_fmt(result)}"
        )

    # ── Trigonometry ─────────────────────────────────────────────────────────
    trig_map = {
        r'sin(?:e)?\s*(?:of\s*)?(\d+\.?\d*)\s*(?:degrees?|°)?':
            ('sin', lambda a: math.sin(math.radians(a))),
        r'cos(?:ine)?\s*(?:of\s*)?(\d+\.?\d*)\s*(?:degrees?|°)?':
            ('cos', lambda a: math.cos(math.radians(a))),
        r'tan(?:gent)?\s*(?:of\s*)?(\d+\.?\d*)\s*(?:degrees?|°)?':
            ('tan', lambda a: math.tan(math.radians(a))),
    }
    for pattern, (name, fn) in trig_map.items():
        m = re.search(pattern, orig)
        if m:
            a = float(m.group(1))
            try:
                result = fn(a)
                return (
                    f"{name.upper()}({_fmt(a)}°)\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Angle    : {_fmt(a)} degrees\n"
                    f"Answer   : {_fmt(result)}"
                )
            except Exception:
                return f"{name.upper()}({a}°) = undefined"

    # ── Logarithm ────────────────────────────────────────────────────────────
    m = re.search(r'log(?:arithm)?\s*(?:of\s*)?(\d+\.?\d*)', orig)
    if m:
        a = float(m.group(1))
        if a <= 0:
            return "Logarithm is undefined for zero or negative numbers."
        return (
            f"Log₁₀({_fmt(a)})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Formula  : log base 10\n"
            f"Answer   : {_fmt(math.log10(a))}"
        )

    m = re.search(r'ln\s*(?:of\s*)?(\d+\.?\d*)', orig)
    if m:
        a = float(m.group(1))
        if a <= 0:
            return "Natural log is undefined for zero or negative numbers."
        return (
            f"ln({_fmt(a)})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Formula  : natural log (base e)\n"
            f"Answer   : {_fmt(math.log(a))}"
        )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def _try_statistics(q: str) -> Optional[str]:
    orig = q.lower()

    # Extract list of numbers
    numbers_match = re.findall(r'-?\d+\.?\d*', q)
    if len(numbers_match) < 2:
        return None
    nums = [float(n) for n in numbers_match]

    if 'mean' in orig or 'average' in orig:
        mean = sum(nums) / len(nums)
        return (
            f"Mean (Average)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Values   : {nums}\n"
            f"Formula  : sum ÷ count\n"
            f"Working  : {sum(nums)} ÷ {len(nums)}\n"
            f"Answer   : {_fmt(mean)}"
        )

    if 'median' in orig:
        sorted_nums = sorted(nums)
        n = len(sorted_nums)
        if n % 2 == 0:
            median = (sorted_nums[n//2 - 1] + sorted_nums[n//2]) / 2
        else:
            median = sorted_nums[n//2]
        return (
            f"Median\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Values sorted: {sorted_nums}\n"
            f"Answer   : {_fmt(median)}"
        )

    if 'mode' in orig:
        from collections import Counter
        count = Counter(nums)
        max_freq = max(count.values())
        modes = [k for k, v in count.items() if v == max_freq]
        return (
            f"Mode\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Values   : {nums}\n"
            f"Answer   : {modes} (appears {max_freq} time{'s' if max_freq > 1 else ''})"
        )

    if 'range' in orig and 'range' not in orig.replace('range', ''):
        r = max(nums) - min(nums)
        return (
            f"Range\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Values   : {nums}\n"
            f"Formula  : max − min\n"
            f"Working  : {max(nums)} − {min(nums)}\n"
            f"Answer   : {_fmt(r)}"
        )

    if 'variance' in orig or 'standard deviation' in orig or 'std' in orig:
        n = len(nums)
        mean = sum(nums) / n
        variance = sum((x - mean) ** 2 for x in nums) / n
        std_dev = math.sqrt(variance)
        return (
            f"Variance & Standard Deviation\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Values   : {nums}\n"
            f"Mean     : {_fmt(mean)}\n"
            f"Variance : {_fmt(variance)}\n"
            f"Std Dev  : {_fmt(std_dev)}"
        )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

def _try_geometry(q: str) -> Optional[str]:
    orig = q.lower()
    nums = [float(n) for n in re.findall(r'\d+\.?\d*', q)]

    # Circle
    if 'circle' in orig:
        if 'area' in orig and nums:
            r = nums[0]
            area = math.pi * r ** 2
            return (
                f"Area of Circle\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Formula  : π × r²\n"
                f"Working  : π × {r}²\n"
                f"         : π × {r**2}\n"
                f"Answer   : {_fmt(area)}"
            )
        if ('circumference' in orig or 'perimeter' in orig) and nums:
            r = nums[0]
            circ = 2 * math.pi * r
            return (
                f"Circumference of Circle\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Formula  : 2 × π × r\n"
                f"Working  : 2 × π × {r}\n"
                f"Answer   : {_fmt(circ)}"
            )

    # Rectangle
    if 'rectangle' in orig and len(nums) >= 2:
        l, w = nums[0], nums[1]
        if 'area' in orig:
            return (
                f"Area of Rectangle\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Formula  : length × width\n"
                f"Working  : {l} × {w}\n"
                f"Answer   : {_fmt(l * w)}"
            )
        if 'perimeter' in orig:
            return (
                f"Perimeter of Rectangle\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Formula  : 2(l + w)\n"
                f"Working  : 2 × ({l} + {w})\n"
                f"Answer   : {_fmt(2 * (l + w))}"
            )

    # Triangle
    if 'triangle' in orig and len(nums) >= 2:
        if 'area' in orig:
            b, h = nums[0], nums[1]
            return (
                f"Area of Triangle\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Formula  : ½ × base × height\n"
                f"Working  : 0.5 × {b} × {h}\n"
                f"Answer   : {_fmt(0.5 * b * h)}"
            )

    # Sphere
    if 'sphere' in orig and nums:
        r = nums[0]
        if 'volume' in orig:
            vol = (4/3) * math.pi * r ** 3
            return (
                f"Volume of Sphere\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Formula  : (4/3) × π × r³\n"
                f"Working  : (4/3) × π × {r}³\n"
                f"Answer   : {_fmt(vol)}"
            )
        if 'surface' in orig:
            sa = 4 * math.pi * r ** 2
            return (
                f"Surface Area of Sphere\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Formula  : 4 × π × r²\n"
                f"Working  : 4 × π × {r}²\n"
                f"Answer   : {_fmt(sa)}"
            )

    # Cylinder
    if 'cylinder' in orig and len(nums) >= 2:
        r, h = nums[0], nums[1]
        if 'volume' in orig:
            vol = math.pi * r ** 2 * h
            return (
                f"Volume of Cylinder\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Formula  : π × r² × h\n"
                f"Working  : π × {r}² × {h}\n"
                f"Answer   : {_fmt(vol)}"
            )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# WORD PROBLEMS
# ─────────────────────────────────────────────────────────────────────────────

def _try_word_problems(q: str) -> Optional[str]:
    orig = q.lower()
    nums = [float(n) for n in re.findall(r'\d+\.?\d*', q)]

    # Profit / Loss
    if ('profit' in orig or 'loss' in orig) and len(nums) >= 2:
        cp, sp = nums[0], nums[1]
        if sp > cp:
            profit = sp - cp
            pct = (profit / cp) * 100
            return (
                f"Profit Calculation\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Cost Price  : {_fmt(cp)}\n"
                f"Sell Price  : {_fmt(sp)}\n"
                f"Profit      : {_fmt(profit)}\n"
                f"Profit %    : {_fmt(pct)}%"
            )
        else:
            loss = cp - sp
            pct = (loss / cp) * 100
            return (
                f"Loss Calculation\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Cost Price  : {_fmt(cp)}\n"
                f"Sell Price  : {_fmt(sp)}\n"
                f"Loss        : {_fmt(loss)}\n"
                f"Loss %      : {_fmt(pct)}%"
            )

    # Speed / Distance / Time
    if 'speed' in orig or 'distance' in orig or 'time' in orig:
        if 'speed' in orig and 'distance' in orig and len(nums) >= 2:
            d, t = nums[0], nums[1]
            speed = d / t
            return (
                f"Speed Calculation\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Formula  : speed = distance ÷ time\n"
                f"Working  : {d} ÷ {t}\n"
                f"Answer   : {_fmt(speed)} units/time"
            )
        if 'distance' in orig and len(nums) >= 2:
            s, t = nums[0], nums[1]
            dist = s * t
            return (
                f"Distance Calculation\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Formula  : distance = speed × time\n"
                f"Working  : {s} × {t}\n"
                f"Answer   : {_fmt(dist)} units"
            )

    # Simple Interest
    if 'simple interest' in orig and len(nums) >= 3:
        p, r, t = nums[0], nums[1], nums[2]
        si = (p * r * t) / 100
        amount = p + si
        return (
            f"Simple Interest\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Formula  : SI = (P × R × T) ÷ 100\n"
            f"Principal: {_fmt(p)}\n"
            f"Rate     : {_fmt(r)}%\n"
            f"Time     : {_fmt(t)} years\n"
            f"SI       : {_fmt(si)}\n"
            f"Amount   : {_fmt(amount)}"
        )

    # Compound Interest
    if 'compound interest' in orig and len(nums) >= 3:
        p, r, t = nums[0], nums[1], nums[2]
        amount = p * (1 + r/100) ** t
        ci = amount - p
        return (
            f"Compound Interest\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Formula  : A = P(1 + r/100)ᵗ\n"
            f"Principal: {_fmt(p)}\n"
            f"Rate     : {_fmt(r)}%\n"
            f"Time     : {_fmt(t)} years\n"
            f"Amount   : {_fmt(amount)}\n"
            f"CI       : {_fmt(ci)}"
        )

    # Ratio
    if 'ratio' in orig and len(nums) >= 2:
        a, b = nums[0], nums[1]
        total = a + b
        frac_a = Fraction(int(a), int(total))
        frac_b = Fraction(int(b), int(total))
        return (
            f"Ratio\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Ratio    : {int(a)} : {int(b)}\n"
            f"Total    : {int(total)} parts\n"
            f"Part A   : {frac_a} of total\n"
            f"Part B   : {frac_b} of total"
        )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# ALGEBRA — linear and quadratic
# ─────────────────────────────────────────────────────────────────────────────

def _try_algebra(q: str) -> Optional[str]:
    orig = q.lower()

    # Linear: ax + b = c  or  ax = b
    m = re.search(r'(-?\d*\.?\d*)\s*x\s*([+\-]\s*\d+\.?\d*)?\s*=\s*(-?\d+\.?\d*)', orig)
    if m:
        a_str = m.group(1).replace(' ', '') or '1'
        a = float(a_str) if a_str not in ('', '-') else (-1.0 if a_str == '-' else 1.0)
        b_str = (m.group(2) or '0').replace(' ', '')
        b = float(b_str) if b_str else 0.0
        c = float(m.group(3))

        if a == 0:
            return None

        x = (c - b) / a
        return (
            f"Linear Equation\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Equation : {m.group(0)}\n"
            f"Step 1   : move constant → {a}x = {c} - ({b}) = {c - b}\n"
            f"Step 2   : divide both sides by {a}\n"
            f"Answer   : x = {_fmt(x)}"
        )

    # Quadratic: ax² + bx + c = 0
    m = re.search(
        r'(-?\d*\.?\d*)\s*x\s*(?:\*\*2|²|\^2)\s*([+\-]\s*\d*\.?\d*\s*x)?\s*([+\-]\s*\d+\.?\d*)?\s*=\s*0',
        orig
    )
    if m:
        a_s = m.group(1).strip() or '1'
        a = float(a_s) if a_s not in ('', '-') else (-1.0 if a_s == '-' else 1.0)
        b_s = re.sub(r'x', '', m.group(2) or '0').replace(' ', '')
        b = float(b_s) if b_s not in ('', '+', '-') else (
            1.0 if b_s == '+' else -1.0 if b_s == '-' else 0.0)
        c_s = (m.group(3) or '0').replace(' ', '')
        c = float(c_s) if c_s else 0.0

        disc = b**2 - 4*a*c
        if disc < 0:
            return (
                f"Quadratic Equation\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Discriminant = {_fmt(disc)} (negative)\n"
                f"Answer   : No real roots (complex roots)"
            )
        x1 = (-b + math.sqrt(disc)) / (2*a)
        x2 = (-b - math.sqrt(disc)) / (2*a)
        return (
            f"Quadratic Equation\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Formula  : x = (-b ± √(b²−4ac)) / 2a\n"
            f"a={_fmt(a)}, b={_fmt(b)}, c={_fmt(c)}\n"
            f"Discriminant: b²−4ac = {_fmt(disc)}\n"
            f"x₁ = {_fmt(x1)}\n"
            f"x₂ = {_fmt(x2)}"
        )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def solve_math_with_explanation(question: str) -> Optional[str]:
    """
    Try every solver in order. Return first non-None result.
    Returns None if no solver can handle it (pass to Llama).
    """
    solvers = [
        _try_special,
        _try_statistics,
        _try_geometry,
        _try_word_problems,
        _try_algebra,
    ]

    for solver in solvers:
        try:
            result = solver(question)
            if result:
                return result
        except Exception:
            continue

    # Last resort: try direct arithmetic eval
    q = _normalise(question)
    q_clean = re.sub(r'[^0-9+\-*/%.() ]', '', q).strip()
    if q_clean:
        result = _safe_eval(q_clean)
        if result is not None:
            return (
                f"Arithmetic\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Expression : {q_clean}\n"
                f"Answer     : {_fmt(result)}"
            )

    return None


if __name__ == "__main__":
    tests = [
        "what is 15% of 240",
        "square root of 144",
        "solve 3x + 5 = 20",
        "find the area of a circle with radius 7",
        "mean of 4, 8, 6, 5, 3",
        "standard deviation of 2, 4, 4, 4, 5, 5, 7, 9",
        "simple interest on 5000 at 8% for 3 years",
        "sin of 45 degrees",
        "volume of a sphere with radius 6",
        "profit if cost price is 200 and selling price is 250",
    ]
    for t in tests:
        print(f"\nQ: {t}")
        print(f"A: {solve_math_with_explanation(t)}")
