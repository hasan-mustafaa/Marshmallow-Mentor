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
        img = Image.open(file.file)
    else:
        print(f"Processing {image_path} for character data...")
        img = Image.open(image_path)
    response = await client.aio.models.generate_content(
        model=model_id,
        contents=[
            (
                "No extra text, no explanations. "
                "The only possible things you are allowed to detect [1,2,3,4,5,6,7,8,9,0,=,x,÷,+, -, /]"
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


async def handwriting_to_latex(image_path: str, file=None) -> str:
    """
    Analyzes an image and returns a single LaTeX string.
    (Async version based on the client pattern from ReturnStringy)
    """
    if file is not None:
        if hasattr(file, "file"):  
            # FastAPI UploadFile
            img = Image.open(file.file)
        else:
            # Already a PIL Image
            img = file  
    else:
        # Load from path
        img = Image.open(image_path)

    # Using the async client pattern from your sample
    response = await client.aio.models.generate_content(
        model=model_id,  # Assuming 'model_id' is in scope, like in your sample
        contents=[
            "You are a perfect handwriting-to-LaTeX OCR. "
            "Convert **only the math** in the image to clean LaTeX. "
            "Return **only** the LaTeX code inside $$ delimiters. "
            "No extra text, no explanations.",
            img,
        ],
        # Using the config object from your sample,
        # but keeping the parameters from the original function
        config=GenerateContentConfig(temperature=0),
    )

    # This post-processing logic is identical to your original function
    text = response.text.strip()
    if "$$" in text:
        start = text.find("$$") + 2
        end = text.rfind("$$")
        latex_body = text[start:end].strip()
        return latex_body  
    else:
        return "There was an error"

async def main():
    image_path = "/Users/hasan/Downloads/image.jpg"
    print(f"Processing image: {image_path}")
    basicpositionlabelling = await ReturnStringy(image_path)
    print(basicpositionlabelling)
    
    #latex = await handwriting_to_latex(image_path)
    #print("LaTeX:", latex)

if __name__ == "__main__":
    asyncio.run(main())

