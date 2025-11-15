from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple, Dict

ErrorType = Literal[
    "correct",
    "parse_error",
    "off_by_one",
    "too_small",
    "too_big",
    "sign_error",
    "carry_error",
    "borrow_error",
    "unknown"
]

TargetType = Literal[
    "lhs", "rhs", "operator",
    "ones_column", "tens_column", "hundreds_column"
]

@dataclass
class Annotation:
    target: TargetType
    message: str

@dataclass
class StepFeedback:
    is_correct: bool
    error_type: ErrorType
    correct_value: Optional[int]
    annotations: List[Annotation]
    debug: Dict[str, object]

def _safe_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except:
        return None

def parse_equation(eq: str):
    eq = eq.replace("×", "*").replace("x", "*").replace("X", "*").replace("÷", "/")
    if "=" not in eq:
        return None, None, None, None

    lhs, rhs = eq.split("=", 1)
    parts = lhs.strip().split()

    if len(parts) != 3:
        return None, None, None, None

    t1 = _safe_int(parts[0])
    op = parts[1]
    t2 = _safe_int(parts[2])
    rhs_student = _safe_int(rhs.strip())

    if t1 is None or t2 is None or rhs_student is None:
        return None, None, None, None
    
    if op not in {"+", "-", "*", "/"}:
        return None, None, None, None

    return t1, op, t2, rhs_student

def compute_correct(term1: int, op: str, term2: int) -> Optional[int]:
    if op == "+":
        return term1 + term2
    if op == "-":
        return term1 - term2
    if op == "*":
        return term1 * term2
    if op == "/":
        if term2 == 0:
            return None
        # only integer division allowed; reject non-integer results
        if term1 % term2 != 0:
            return None
        return term1 // term2
    return None


def _digits(n: int) -> List[int]:
    n = abs(n)
    return [n // 1 % 10, n // 10 % 10, n // 100 % 10]

def detect_carry(term1, term2, student, correct) -> bool:
    s = _digits(student)
    c = _digits(correct)
    a = _digits(term1)
    b = _digits(term2)

    if a[0] + b[0] >= 10:
        if s[0] == (a[0] + b[0]) % 10 and s[1] != c[1]:
            return True
    return False

def detect_borrow(term1, term2, student, correct) -> bool:
    s = _digits(student)
    c = _digits(correct)
    a = _digits(term1)
    b = _digits(term2)

    if a[0] < b[0]:
        no_borrow_digit = (a[0] - b[0]) % 10
        if s[0] == no_borrow_digit and s[1] != c[1]:
            return True
    return False

def analyze_step(equation: str) -> StepFeedback:
    term1, op, term2, rhs_student = parse_equation(equation)

    if term1 is None:
        return StepFeedback(False, "parse_error", None,
            [Annotation("lhs", "Format should be like: 73 + 28 = 90")],
            {"raw": equation}
        )

    rhs_correct = compute_correct(term1, op, term2)
    if rhs_correct is None:
        return StepFeedback(False, "parse_error", None,
            [Annotation("operator", "Invalid or non-integer result")],
            {}
        )
    
    if rhs_student == rhs_correct:
        return StepFeedback(True, "correct", rhs_correct,
            [Annotation("rhs", "Correct!")],
            {}
        )

    if abs(rhs_student - rhs_correct) == 1:
        return StepFeedback(False, "off_by_one", rhs_correct,
            [Annotation("rhs", "So close! Off by 1.")],
            {}
        )

    if op in {"+", "-"}:
        wrong_op = "-" if op == "+" else "+"
        alt = compute_correct(term1, wrong_op, term2)
        if alt == rhs_student:
            return StepFeedback(False, "sign_error", rhs_correct,
                [
                    Annotation("operator", f"Used '{wrong_op}' instead of '{op}'."),
                    Annotation("rhs", "Try again using the correct sign.")
                ],
                {}
            )

    if op == "+" and detect_carry(term1, term2, rhs_student, rhs_correct):
        return StepFeedback(False, "carry_error", rhs_correct,
            [
                Annotation("ones_column", "Check the ones column – carrying needed."),
                Annotation("tens_column", "The tens column changes when carrying.")
            ],
            {}
        )

    if op == "-" and detect_borrow(term1, term2, rhs_student, rhs_correct):
        return StepFeedback(False, "borrow_error", rhs_correct,
            [
                Annotation("ones_column", "You needed to borrow here."),
                Annotation("tens_column", "Borrowing affects the tens column.")
            ],
            {}
        )

    if rhs_student < rhs_correct:
        return StepFeedback(False, "too_small", rhs_correct,
            [Annotation("rhs", "Your answer is too small.")],
            {}
        )

    if rhs_student > rhs_correct:
        return StepFeedback(False, "too_big", rhs_correct,
            [Annotation("rhs", "Your answer is too big.")],
            {}
        )

    # fallback
    return StepFeedback(False, "unknown", rhs_correct,
        [Annotation("rhs", "Something is off. Try again.")],
        {}
    )

def process_equation(equation_str: str, tokens, column_boxes=None):
    feedback = analyze_step(equation_str)
    return feedback

