import os
import json
from typing import List
from PIL import Image
from dotenv import load_dotenv
from google.genai.types import GenerateContentConfig, HttpOptions
from google import genai
import asyncio


class CharacterData:
    def __init__(self, label: str, box: List[List[int]]) -> None:
        self.label = label
        self.box = box


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment or .env file!")
client = genai.Client(
    api_key=GEMINI_API_KEY, http_options=HttpOptions(api_version="v1beta")
)
model_id = "gemini-2.5-flash"  # Using 1.5-flash, as 2.5 is not a public model ID


async def ReturnStringy(image_path: str, file=None) -> List[CharacterData]:
    """
    Analyzes an image and returns a list of CharacterData objects,
    each with a label and bounding box.
    """
    if file:
        img = file
    else:
        print(f"Processing {image_path} for character data...")
        img = Image.open(image_path)
    response = await client.aio.models.generate_content(
        model=model_id,
        contents=[
            (
                "No extra text, no explanations. "
                "Define position [0,0] as the bottom left corner of the image. "
                "First index of position is x and second one is y. "
                "For each character, assign it with an array [topleft, topright, leftbottom, rightbottom]. "
                "These are the corners in which the character lays over. "
                'Something like [{ "label": "2", "box": [[0,20],[20,20],[0,0],[20,0]]}]'
                "RETURN A VALID JSON LIST OF OBJECTS!!! DO NOT GIVE '''JSON, give raw json"
            ),
            img,
        ],
        config=GenerateContentConfig(temperature=0),
    )
    content = response.text.strip()
    print(content)
    content = json.loads(content)
    print(content)
    data: List[CharacterData] = []

    for resp in content:
        neobj = CharacterData(resp["label"], resp["box"])
        data.append(neobj)

    return data  # Return the list of objects


def handwriting_to_latex(image_path: str) -> str:
    img = Image.open(image_path)
    response = model.generate_content(
        [
            "You are a perfect handwriting-to-LaTeX OCR. "
            "Convert **only the math** in the image to clean LaTeX. "
            "Return **only** the LaTeX code inside $$ delimiters. "
            "No extra text, no explanations.",
            img,
        ],
        generation_config=genai.GenerationConfig(
            temperature=0, response_mime_type="text/plain"
        ),
    )
    text = response.text.strip()
    if "$$" in text:
        start = text.find("$$") + 2
        end = text.rfind("$$")
        latex_body = text[start:end].strip()
        return (
            "$" + latex_body + "$"
        )  # Forces inline math latex, as gemini returns without dollar signs
    else:
        return "There was an error"


def ForEachReturnError(str) -> str:
    img = Image.open(image_path)
    response = model.generate_content(
        [
            "In this json response, a math problem is inscribed. This includes fractions, arthmetic."
            "transcribe the following problem into an equation"
            "if error is present for the label that is incorrect, put error=True, expectedresult={answer}"
            'For example,your input would look somehting like this: { "label": "2", "box": [[0,20],[20,20] ,[0,0] [20, 0] }',
            'Your output would look something like { "label": "2", "box": [[0,20],[20,20] ,[0,0] [20, 0] , "error":True, "expectedresult=4}',
            img,
        ],
        generation_config=genai.GenerationConfig(
            temperature=0, response_mime_type="text/plain"
        ),
    )
    return response.text


async def main():
    image_path = "/home/dilyxs/Downloads/sample_let.png"
    print(f"Processing image: {image_path}")
    basicpositionlabelling = await ReturnStringy(image_path)
    print(basicpositionlabelling)


if __name__ == "main":
    asyncio.run(main())
