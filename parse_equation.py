from sympy import symbols, N
from sympy.parsing.latex import parse_latex

def parse_equation_v2(latex_string: str):
    # Format Latex Equation
    equation = latex_string.strip().replace(" ", "")
    left_side, right_side = equation.split("=", 1)

    # Parse both sides properly
    left_expression = parse_latex(left_side)
    right_expression = parse_latex(right_side)

    # Evaluate it numerically
    left_value = N(left_expression)
    right_value = N(right_expression)

    # Check if the equation is correct
    is_correct = left_value == right_value

    return left_expression, right_expression, left_value, right_value, is_correct