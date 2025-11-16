import os
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv


async def ocr_data_to_latex(ocr_data: list) -> str:
    load_dotenv()
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    client = AsyncOpenAI(
        api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1"
    )

    """
    Converts a list of OCR'd objects (with 'label' and 'box' keys) 
    to a LaTeX string by prompting an LLM to analyze spatial relationships.
    """
    input_json_str = json.dumps(ocr_data, indent=2)
    system_prompt = (
        "You are an expert mathematical typesetter. "
        "You will be given a JSON list of OCR-detected characters, "
        "each with a 'label' (the character) and a 'box' (its coordinates).\n\n"
        "Your task is to analyze the **spatial relationships** of the bounding boxes "
        "to correctly interpret the full mathematical expression.\n\n"
        "Use the coordinates to identify structures like:\n"
        "- Horizontal numbers and variables (e.g., '1' and '2' becoming '12')\n"
        "- Fractions (vertical stacking)\n"
        "- Superscripts (exponents)\n"
        "- Subscripts\n"
        "- Limits, integrals, and summations\n\n"
        "Return **only** the single, complete LaTeX string, "
        "enclosed in `$$ ... $$` delimiters. "
        "Do not include any other text, explanations, or markdown."
    )

    try:
        chat_completion = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_json_str},
            ],
            temperature=0.0,  # Math conversion should be deterministic
            max_tokens=1024,
        )

        text = chat_completion.choices[0].message.content.strip()

        if "$$" in text:
            start = text.find("$$") + 2
            end = text.rfind("$$")
            if end > start:
                latex_body = text[start:end].strip()
                return f"${latex_body}$"
        if text and not text.startswith("Error"):
            cleaned_text = text.replace("`", "").strip()
            return f"${cleaned_text}$"

        return "Error: Could not extract LaTeX."

    except Exception as e:
        return f"Error: API call failed. {e}"


import asyncio

data = [
    {"label": "2", "box": [[250, 3050], [350, 3050], [250, 2900], [350, 2900]]},
    {"label": "3", "box": [[380, 3050], [480, 3050], [380, 2900], [480, 2900]]},
    {"label": "-", "box": [[550, 2980], [620, 2980], [550, 2960], [620, 2960]]},
    {"label": "1", "box": [[680, 3050], [730, 3050], [680, 2900], [730, 2900]]},
    {"label": "2", "box": [[780, 3050], [880, 3050], [780, 2900], [880, 2900]]},
    {"label": "=", "box": [[950, 3000], [1030, 3000], [950, 2940], [1030, 2940]]},
    {"label": "1", "box": [[1150, 3050], [1200, 3050], [1150, 2900], [1200, 2900]]},
    {"label": "0", "box": [[1250, 3050], [1350, 3050], [1250, 2900], [1350, 2900]]},
]

data_fraction = [
    {"label": "-", "box": [[50, 275], [250, 275], [50, 225], [250, 225]]},
    {"label": "2", "box": [[100, 400], [200, 400], [100, 300], [200, 300]]},  # Bottom
]


async def main():
    print("Testing simple expression:")
    sorted_data = sorted(data, key=lambda item: item["box"][0][0])
    latex_result = await ocr_data_to_latex(sorted_data)
    print(f"Input: {data[0]['label']}...")
    print(f"Output: {latex_result}")  # Expected: $23 - 12 = 10$

    print("\nTesting fraction expression:")
    latex_fraction = await ocr_data_to_latex(data_fraction)
    print(f"Input: {data_fraction[0]['label']}...")
    print(f"Output: {latex_fraction}")  # Expected: $\frac{1}{2}$


if __name__ == "__main__":
    asyncio.run(main())
