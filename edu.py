import os
import json
from io import BytesIO
from typing import List, Optional, Literal, Tuple, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathlib import Path
from PIL import Image
import google.generativeai as genai

from image_recognition import ReturnStringy
from parse_equation import parse_equation_v2
from voice_model import generate_code, generate_hint

# gemini config

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

GEMINI_API_KEY: Optional[str] = None
try:
    with open(ENV_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("GEMINI_API_KEY="):
                GEMINI_API_KEY = line.split("=", 1)[1].strip()
                break
except FileNotFoundError:
    GEMINI_API_KEY = None

print("DEBUG GEMINI_API_KEY loaded:", bool(GEMINI_API_KEY))

if not GEMINI_API_KEY:
    raise ValueError(f"GEMINI_API_KEY not found in {ENV_PATH} or environment!")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# error types

ErrorType = Literal[
    "correct",
    "parse_error",
    "off_by_one",
    "too_small",
    "too_big",
    "sign_error",
    "carry_error",
    "borrow_error",
    "unknown",
]

TargetType = Literal[
    "lhs", "rhs", "operator", "ones_column", "tens_column", "hundreds_column"
]


class Annotation:
    def __init__(self, target: TargetType, message: str):
        self.target = target
        self.message = message


class StepFeedback:
    def __init__(
        self,
        is_correct: bool,
        error_type: ErrorType,
        correct_value: Optional[int],
        annotations: List[Annotation],
        debug: Dict[str, object],
    ):
        self.is_correct = is_correct
        self.error_type = error_type
        self.correct_value = correct_value
        self.annotations = annotations
        self.debug = debug


# error logic


def _safe_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except:
        return None


def parse_equation(eq: str):
    """Parse strings like '73 + 28 = 90' into (term1, op, term2, rhs_student)."""
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
    "Compute the correct result only for integer division"
    if op == "+":
        return term1 + term2
    if op == "-":
        return term1 - term2
    if op == "*":
        return term1 * term2
    if op == "/":
        if term2 == 0:
            return None
        if term1 % term2 != 0:
            return None
        return term1 // term2
    return None


def _digits(n: int) -> List[int]:
    "Return [ones, tens, hundreds] of n"
    n = abs(n)
    return [n // 1 % 10, n // 10 % 10, n // 100 % 10]


def detect_carry(term1, term2, student, correct) -> bool:
    "addition carry mistake"
    s = _digits(student)
    c = _digits(correct)
    a = _digits(term1)
    b = _digits(term2)

    if a[0] + b[0] >= 10:
        if s[0] == (a[0] + b[0]) % 10 and s[1] != c[1]:
            return True
    return False


def detect_borrow(term1, term2, student, correct) -> bool:
    "forget to borrow in subtraction"
    s = _digits(student)
    c = _digits(correct)
    a = _digits(term1)
    b = _digits(term2)

    if a[0] < b[0]:
        no_borrow_digit = (a[0] - b[0]) % 10
        if s[0] == no_borrow_digit and s[1] != c[1]:
            return True
    return False


class StepFeedback:
    def __init__(
        self,
        is_correct: bool,
        error_type: ErrorType,
        correct_value: Optional[int],
        annotations: List[Annotation],
        debug: Dict[str, object],
    ):
        self.is_correct = is_correct
        self.error_type = error_type
        self.correct_value = correct_value
        self.annotations = annotations
        self.debug = debug


async def analyze_step(equation: str) -> StepFeedback:
    """
    Error detector:
      - parse_error
      - correct
      - off_by_one
      - sign_error
      - carry_error
      - borrow_error
      - too_small / too_big
      - unknown
    """
    left_expression, right_expression, left_value, right_value, is_correct = (
        parse_equation_v2(equation)
    )
    input = f"equation: {equation}, left_expression: {left_expression}, right_expression: {right_expression}, left_value: {left_value}, right_value: {right_value}, is_correct: {is_correct}"
    response = await generate_code(
        (
            "You are a model that returns a JSON analysis of a math equation. "
            "You must return a JSON with the exact following format: "
            '{"is_correct": <bool>, "error_type": "<str>", "correct_value": <int or null>, '
            '"annotations": ["<string hint 1>", "<string hint 2>"], '
            '"debug": {"explanation": "<string explanation>", "solution": "<string solution>"}}. '
            "SEND it as a pure {} DO NOT INCLUDE three commas or JSON"
        ),
        input,
    )
    response_dict = json.loads(response)
    return StepFeedback(
        is_correct=response_dict["is_correct"],
        error_type=response_dict["error_type"],
        correct_value=response_dict.get("correct_value"),
        annotations=response_dict.get("annotations", []),  # Use .get for robustness
        debug=response_dict.get("debug", {}),
    )


# 3. DYNAMIC AR TARGET SELECTION FROM TOKENS


def _center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _height(bbox: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return abs(y2 - y1)


def cluster_rows(tokens: List[Dict]) -> List[List[Dict]]:
    """
    Group tokens into rows based on vertical (y) positions.
    Works even for messy or bad handwriting.
    """
    if not tokens:
        return []

    tokens_sorted = sorted(tokens, key=lambda t: _center(t["bbox"])[1])

    heights = [_height(t["bbox"]) for t in tokens_sorted]
    avg_h = sum(heights) / len(heights) if heights else 1.0
    threshold = avg_h * 0.6  # tweakable
    rows: List[List[Dict]] = []

    for t in tokens_sorted:
        cx, cy = _center(t["bbox"])
        if not rows:
            rows.append([t])
        else:
            last_row = rows[-1]
            _, last_cy = _center(last_row[0]["bbox"])
            if abs(cy - last_cy) <= threshold:
                last_row.append(t)
            else:
                rows.append([t])

    for row in rows:
        row.sort(key=lambda t: _center(t["bbox"])[0])

    return rows


def find_layout(tokens: List[Dict]) -> Dict[str, Optional[Dict]]:
    """
    tokens from bbox:
      - operator_token
      - answer_row_digits
      - ones_token, tens_token, hundreds_token
      - rhs_token
    """
    rows = cluster_rows(tokens)
    layout: Dict[str, Optional[Dict]] = {
        "operator_token": None,
        "answer_row_digits": None,
        "ones_token": None,
        "tens_token": None,
        "hundreds_token": None,
        "rhs_token": None,
    }

    if not rows:
        return layout

    # operator anywhere
    for t in tokens:
        if t.get("text") in {"+", "-", "*", "/"}:
            layout["operator_token"] = t
            break

    # assume last row is answer row
    answer_row = rows[-1]
    digit_tokens = [t for t in answer_row if t["text"].isdigit()]
    layout["answer_row_digits"] = digit_tokens

    if not digit_tokens:
        return layout

    digit_tokens.sort(key=lambda t: _center(t["bbox"])[0])

    ones_token = digit_tokens[-1]
    layout["ones_token"] = ones_token
    layout["rhs_token"] = ones_token

    if len(digit_tokens) > 1:
        layout["tens_token"] = digit_tokens[-2]
    if len(digit_tokens) > 2:
        layout["hundreds_token"] = digit_tokens[-3]

    return layout


def pick_target_point_from_tokens(
    feedback: StepFeedback, tokens: List[Dict]
) -> Tuple[float, float]:
    """
    Decide where the AR character should go based on:
      - feedback from error types
      - tokens with bbox from OCR
    """
    if not tokens:
        return (0.0, 0.0)

    layout = find_layout(tokens)
    op_tok = layout["operator_token"]
    ones_tok = layout["ones_token"]
    tens_tok = layout["tens_token"]
    rhs_tok = layout["rhs_token"]

    def avg_center_all():
        xs, ys = [], []
        for t in tokens:
            cx, cy = _center(t["bbox"])
            xs.append(cx)
            ys.append(cy)
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    # parse error
    if feedback.error_type == "parse_error":
        first = sorted(
            tokens, key=lambda t: (_center(t["bbox"])[1], _center(t["bbox"])[0])
        )[0]
        return _center(first["bbox"])

    # correct
    if feedback.error_type == "correct":
        if rhs_tok is not None:
            return _center(rhs_tok["bbox"])
        return avg_center_all()

    # sign error
    if feedback.error_type == "sign_error":
        if op_tok is not None:
            return _center(op_tok["bbox"])
        return avg_center_all()

    # carry
    if feedback.error_type in {"carry_error", "borrow_error"}:
        if ones_tok is not None:
            return _center(ones_tok["bbox"])
        if tens_tok is not None:
            return _center(tens_tok["bbox"])
        if rhs_tok is not None:
            return _center(rhs_tok["bbox"])
        return avg_center_all()

    # other errors
    if rhs_tok is not None:
        return _center(rhs_tok["bbox"])
    return avg_center_all()


def process_equation_with_tokens(
    equation_str: str, tokens: List[Dict]
) -> Tuple[StepFeedback, Tuple[float, float]]:
    """
    Full pipeline:
      - run error logic
      - choose AR target point from bounding boxes
    """
    feedback = analyze_step(equation_str)
    target_point = pick_target_point_from_tokens(feedback, tokens)
    return feedback, target_point


# 4. grid generator for ar walk path


def make_grid(
    image_width: int, image_height: int, rows: int, cols: int
) -> List[List[Dict[str, float]]]:
    """
    Split the image into a rows x cols grid.
    Return a 2D array (list of lists) of center points
    """
    grid: List[List[Dict[str, float]]] = []
    cell_w = image_width / cols
    cell_h = image_height / rows

    for r in range(rows):
        row: List[Dict[str, float]] = []
        for c in range(cols):
            center_x = (c + 0.5) * cell_w
            center_y = (r + 0.5) * cell_h
            row.append({"x": center_x, "y": center_y})
        grid.append(row)

    return grid


# 5. gemini: image to latex to eq


def handwriting_image_to_latex(img: Image.Image) -> str:
    """
    Send an image to Gemini and get back latex
    """
    response = gemini_model.generate_content(
        [
            "You are a perfect handwriting-to-LaTeX OCR. "
            "Convert ONLY the math in the image to clean LaTeX. "
            "Return ONLY the LaTeX code inside $$ delimiters. "
            "No extra text, no explanations.",
            img,
        ],
        generation_config=genai.GenerationConfig(
            temperature=0, response_mime_type="text/plain"
        ),
    )

    text = (response.text or "").strip()
    if "$$" in text:
        start = text.find("$$") + 2
        end = text.rfind("$$")
        latex_body = text[start:end].strip()
        return "$" + latex_body + "$"
    else:
        return text


def latex_to_equation(latex: str) -> str:
    "Strip $ and spaces from eq"
    if not latex:
        return ""
    return latex.replace("$", "").strip()


# 6. fastapi models


class AnnotationDTO(BaseModel):
    target: str
    message: str


class TokenDTO(BaseModel):
    text: str
    bbox: Tuple[float, float, float, float]


class AnalyzeEquationRequest(BaseModel):
    equation_str: str
    tokens: List[TokenDTO]
    image_width: int
    image_height: int


class AnalyzeImageResponse(BaseModel):
    equation_str: str
    error_type: str
    is_correct: bool
    analyzeEquationOBJ: AnalyzeEquationRequest
    correct_value: int | None
    datapoints: List[TokenDTO]
    annotations: List[AnnotationDTO]


class AnalyzeEquationResponse(BaseModel):
    equation_str: str
    error_type: str
    is_correct: bool
    correct_value: int | None
    annotations: List[AnnotationDTO]
    target_point: Dict[str, float]
    grid_points: List[List[Dict[str, float]]]  # 2D grid for walking around
    image_width: int
    image_height: int


class HintsRequest(BaseModel):
    error_message: str
    language: str


class HintsResponse(BaseModel):
    response: str


# 7. Fast api apps

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def function():
    return {"message": "true"}


@app.post("/", response_model=HintsResponse)
async def returnHints(req: HintsRequest):
    try:
        data = await generate_hint(req.error_message, req.language)
        HintsResponse(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"incorrect hints as {e}")


@app.post("/analyze-image", response_model=AnalyzeImageResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Unity sends an image
    Backend:
      1) image -> Latex (Gemini)
      2) Latex -> equation
      3) equation -> error feedback thru analysis
    """
    try:
        data = await file.read()
        if not data:
            raise ValueError("Empty file")
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        latex = handwriting_image_to_latex(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini OCR failed: {e}")

    equation_str = latex_to_equation(latex)
    if not equation_str:
        raise HTTPException(
            status_code=422, detail="Could not extract equation from image"
        )

    feedback = analyze_step(equation_str)
    data = []
    datapoints = await ReturnStringy("", file)
    for datapoint in datapoints:
        label = datapoint.label
        points = datapoint.box
        x1 = min([i[0] for i in points])
        y1 = min([i[1] for i in points])
        x2 = max([i[0] for i in points])
        y2 = max([i[1] for i in points])
        newobject = TokenDTO(text=label, bbox=[x1, y1, x2, y2])
        data.append(newobject)

    RequestObg = AnalyzeEquationRequest(
        equation_str=equation_str,
        tokens=data,
        image_width=img.width,
        image_height=img.height,
    )
    return AnalyzeImageResponse(
        equation_str=equation_str,
        analyzeEquationOBJ=RequestObg,
        error_type=feedback.error_type,
        is_correct=feedback.is_correct,
        correct_value=feedback.correct_value,
        datapoints=data,
        annotations=[
            AnnotationDTO(target=a.target, message=a.message)
            for a in feedback.annotations
        ],
    )


@app.post("/analyze-equation", response_model=AnalyzeEquationResponse)
async def analyze_equation(req: AnalyzeEquationRequest):
    """
    Unity sends:
      - equation
      - tokens: [{text, bbox}], where bbox is in image coordinates
      - image width / image height

    Backend:
      - runs error logic
      - picks AR target_point (x,y) from bounding boxes
      - also builds a 2x2 grid of points (for walking to 4 corners)
      - returns everything
    """
    tokens = [{"text": t.text, "bbox": t.bbox} for t in req.tokens]
    feedback, (tx, ty) = process_equation_with_tokens(req.equation_str, tokens)

    # 2x2 grid → 4 "corner-ish" positions
    grid = make_grid(req.image_width, req.image_height, rows=2, cols=2)

    return AnalyzeEquationResponse(
        equation_str=req.equation_str,
        error_type=feedback.error_type,
        is_correct=feedback.is_correct,
        correct_value=feedback.correct_value,
        annotations=[
            AnnotationDTO(target=a.target, message=a.message)
            for a in feedback.annotations
        ],
        target_point={"x": tx, "y": ty},
        grid_points=grid,
        image_width=req.image_width,
        image_height=req.image_height,
    )
