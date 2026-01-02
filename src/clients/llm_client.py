import asyncio

import httpx
from loguru import logger

from src.core.config import GEMINI_API_KEY, GEMINI_RATE_LIMIT_DELAY, HTTP_TIMEOUT, MODEL_NAME

timeout = httpx.Timeout(HTTP_TIMEOUT)
client = httpx.AsyncClient(timeout=timeout)


async def call_llm(prompt: str) -> str:
    """
        Send a prompt to the Gemini LLM and return the generated response text.

        Handles rate limiting, basic response validation, and error cases.
        """
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API Key not found")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 100,
        },
    }

    try:
        response = await client.post(url, json=payload)

        if response.status_code == 429:
            logger.warning("Rate limit hit (429). Waiting 5s...")
            await asyncio.sleep(GEMINI_RATE_LIMIT_DELAY)
            return "1"

        response.raise_for_status()
        result = response.json()

        candidates = result.get("candidates", [])

        if not candidates:
            logger.error(f"API returned no candidates. Full response: {result}")
            return "1"

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])

        if not parts:
            finish_reason = candidates[0].get("finishReason")
            logger.warning(f"No parts in response. Finish reason: {finish_reason}")
            return "1"

        return parts[0].get("text", "1").strip()

    except httpx.HTTPStatusError as e:
        return f"Error HTTP: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Unexpected error occurred: {str(e)}"
