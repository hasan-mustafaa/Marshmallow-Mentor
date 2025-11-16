#Undertand how this works + fine tune voice model

import os
import requests
from dotenv import load_dotenv

# === DEBUG: Show current directory ===
print("Current working directory:", os.getcwd())
print("Looking for .env in this folder...")

# Load .env from the same directory as this script
env_path = os.path.join(os.path.dirname(__file__), '.env')
print(f"Loading .env from: {env_path}")

load_dotenv(env_path)

# === CONFIGURATION ===
API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not API_KEY:
    print("ERROR: ELEVENLABS_API_KEY not found!")
    print("Make sure:")
    print("  1. .env file is in the same folder as this script")
    print("  2. It contains: ELEVENLABS_API_KEY=your_key_here")
    print("  3. No quotes, no spaces around =")
    raise SystemExit

print(f"API Key loaded: {API_KEY[:10]}...{API_KEY[-4:]}")  # Show partial key

# === REST OF YOUR CODE ===
VOICE_ID = "56AoDkrOh6qfVPDXZ7Pt"

VOICE_SETTINGS = {
    "stability": 0.4,          # Lower = more expressive/wild (cartoony variation)
    "similarity_boost": 0.8,   # Keeps it true to the voice's charm
    "style": 0.7,              # Higher = more dramatic/exaggerated (cartoon flair)
    "speed": 1.2               # Slightly faster = energetic, kid-like bounce
}

TEXT = "你好，我叫棉花糖，我是你的AI导师"
OUTPUT_FILE = "output.mp3"


URL = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
headers = {
    "xi-api-key": API_KEY,
    "Content-Type": "application/json"
}
data = {
    "text": TEXT,
    "model_id": "eleven_multilingual_v2"
}


# === TTS REQUEST ===
data = {
    "text": TEXT,
    "model_id": "eleven_multilingual_v2",
    "voice_settings": {
        "stability": 0.4,
        "similarity_boost": 0.8,
        "style": 0.7,
        "speed": 1.2,
        "language": LANG_CODE    # << IMPORTANT LINE
    }
}


print("Sending request to ElevenLabs...")
response = requests.post(URL, json=data, headers=headers)

if response.status_code == 200:
    with open(OUTPUT_FILE, "wb") as f:
        f.write(response.content)
    print(f"SUCCESS! Saved as '{OUTPUT_FILE}'")
else:
    print(f"API Error {response.status_code}: {response.text}")
