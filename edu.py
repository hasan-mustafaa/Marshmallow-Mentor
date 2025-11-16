import os
from voice_model import text_to_speech
import uuid
import json
from io import BytesIO
from typing import List, Optional, Literal, Tuple, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathlib import Path
from PIL import Image
import google.generativeai as genai

from image_recognition import ReturnStringy, handwriting_to_latex
from parse_equation import parse_equation_v2
from voice_model import generate_code, generate_hint
from fastapi.staticfiles import StaticFiles

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


class Annotation:
    def __init__(self, target: str, message: str):
        self.target = target
        self.message = message


class StepFeedback:
    def __init__(
        self,
        is_correct: bool,
        error_type: str,
        correct_value: Optional[int],
        annotations: List[str],
        debug: Dict[str, str],
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
        annotations=response_dict.get("annotations"),  # Use .get for robustness
        debug=response_dict.get(
            "debug",
        ),
    )


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
    annotations: List[str]


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

app.mount("/audio", StaticFiles(directory="static/audio"), name="audio_files")
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


@app.post("/customvoice", response_model=HintsResponse)
async def getmessage(req: HintsResponse):
    try:
        baseurl = "http://148.230.90.188:8000/"
        id = uuid.uuid4()
        await text_to_speech(req.response, f"./static/audio/{id}_audio.mp3")
        public_url = f"{baseurl}audio/{id}_audio.mp3"
        return HintsResponse(response=public_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"incorrect format as {e}")


@app.post("/hints", response_model=HintsResponse)
async def returnHints(req: HintsRequest):
    try:
        data = await generate_hint(req.error_message, req.language)
        ourobject = HintsResponse(response=data)
        return ourobject
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
        original_img = Image.open(BytesIO(data))
        original_format = original_img.format
        img = original_img.convert("RGB")
        if not original_format:
            file_extension = (
                "jpg"  # Defaulting to jpg, but you could also raise an error
            )
        else:
            file_extension = original_format.lower()
        if file_extension == "jpeg":
            file_extension = "jpg"
        base_filename = str(uuid.uuid4())
        save_path = f"./{base_filename}.{file_extension}"
        print(save_path)

        img.save(save_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    equation_latex = await handwriting_to_latex("", img)
    feedback = await analyze_step(equation_latex)
    print(feedback)
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
        equation_str=equation_latex,
        tokens=data,
        image_width=img.width,
        image_height=img.height,
    )
    return AnalyzeImageResponse(
        equation_str=equation_latex,
        analyzeEquationOBJ=RequestObg,
        error_type=feedback.error_type,
        is_correct=feedback.is_correct,
        correct_value=feedback.correct_value,
        datapoints=data,
        annotations=feedback.annotations,
    )
