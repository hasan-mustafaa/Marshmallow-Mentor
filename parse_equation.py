from sympy import symbols, N
from sympy.parsing.latex import parse_latex

latex_string = latex_string = r"3\left(\frac{3}{4} + \frac{5}{6}\right) = \frac{19}{12} - 1"
def parse_equation(latex_string: str):
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

    print("Left expr:", left_expression)
    print("Right expr:", right_expression)
    print("Left value:", left_value)
    print("Right value:", right_value)
    print("Correct?", is_correct)
    

parse_equation(latex_string)
