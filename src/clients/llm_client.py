import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from src.core.config import MODEL_NAME

current_file = Path(__file__)
env_path = current_file.parent.parent / "testing" / ".env"
load_dotenv(dotenv_path=env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

timeout = httpx.Timeout(200.0)
client = httpx.AsyncClient(timeout=timeout)

async def call_llm(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API Key not found")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2048,
        }
    }

    try:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        result = response.json()

        candidates = result.get("candidates", [])
        if not candidates:
            return "Error: The model did not return any responses (content blocking possible)."

        return candidates[0]["content"]["parts"][0]["text"].strip()

    except httpx.HTTPStatusError as e:
        return f"Error HTTP: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Unexpected error occurred: {str(e)}"
