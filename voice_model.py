# marshmallow_mentor.py
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

# Use customised error message parse 

# ------------------- ENV -------------------
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not ELEVENLABS_API_KEY or not GROQ_API_KEY:
    raise SystemExit("Missing API keys in .env")

# ------------------- SETTINGS -------------------
language = "chinese" 
VOICE_ID = "56AoDkrOh6qfVPDXZ7Pt"
OUTPUT_FILE = "output.mp3"
VOICE_SETTINGS = {
    "stability": 0.4,
    "similarity_boost": 0.8,
    "style": 0.7,
    "speed": 1.2,
}    

# ------------------- LLM CLIENT -------------------
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# ------------------- FUNCTION 1: Generate Hint -------------------
def generate_hint(error_message, language):
    chat_completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Marshmallow, a friendly AI math and STEM mentor for kids "
                    "who gives short, encouraging 2–3 line hints in the requested language!"
                )
            },
            {
                "role": "user",
                "content": (
                    f'The user got this error: "{error_message}". '
                    f"Give a 2–3 line hint in {language}."
                ),
            }
        ],
    )
    return chat_completion.choices[0].message.content

# ------------------- TEXT TO SPEECH -------------------
def text_to_speech(text: str, output_file: str = OUTPUT_FILE) -> None:
    """Send text to ElevenLabs and save as MP3."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": VOICE_SETTINGS  # ** unpacking not needed here
    }

    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"Audio saved: {output_file}")
    else:
        raise SystemExit(f"ElevenLabs Error {response.status_code}: {response.text}")


# ------------------- MAIN -------------------
if __name__ == "__main__":
   # Example placeholder — replace when calling your function
    error_message = "NameError: variable not defined" # Use customised error message parse 

    #Generate hint and convert to speech
    hint = generate_hint(error_message, language)
    text_to_speech(hint)


