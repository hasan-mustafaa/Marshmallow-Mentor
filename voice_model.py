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

if not ELEVENLABS_API_KEY:
    raise SystemExit("ERROR: ELEVENLABS_API_KEY missing in .env")
if not GROQ_API_KEY:
    raise SystemExit("ERROR: GROQ_API_KEY missing in .env")

# ------------------- SETTINGS -------------------
language = "english"     # <<< your requested variable

# ------------------- LLM CLIENT -------------------
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

VOICE_ID = "56AoDkrOh6qfVPDXZ7Pt"

VOICE_SETTINGS = {
    "stability": 0.4,
    "similarity_boost": 0.8,
    "style": 0.7,
    "speed": 1.2,
}

# Example placeholder — replace when calling your function
error_message = "NameError: variable not defined" # Use customised error message parse 


# ------------------- LLM COMPLETION -------------------
chat_completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": (
                "You are Marshmallow, a friendly AI math and stem mentor for kids who gives "
                "short, encouraging 2–3 line hints in the requested language!"
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

TEXT = chat_completion.choices[0].message.content

# ------------------- TEXT TO SPEECH -------------------
OUTPUT_FILE = "output.mp3"

URL = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

headers = {
    "xi-api-key": ELEVENLABS_API_KEY,
    "Content-Type": "application/json"
}

data = {
    "text": TEXT,
    "model_id": "eleven_multilingual_v2",
    "voice_settings": {
        **VOICE_SETTINGS,
    }
}

response = requests.post(URL, json=data, headers=headers)

if response.status_code == 200:
    with open(OUTPUT_FILE, "wb") as f:
        f.write(response.content)
else:
    raise SystemExit(f"ElevenLabs API Error {response.status_code}: {response.text}")
