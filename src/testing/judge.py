import re
from typing import Optional

import anthropic
from loguru import logger

from src.core.config import CLAUDE_MAX_TOKENS, CLAUDE_MODEL

client = anthropic.Anthropic()


def clean_json(response_text: str) -> str:
    """
    Normalize model output by removing Markdown code blocks.

    Args:
        response_text: Raw response text from the model.

    Returns:
        Cleaned JSON string.
    """
    pattern = r"```(?:json)?\s*(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL)

    if match:
        return match.group(1).strip()
    return response_text.strip()


def judge_answer(prompt: str) -> Optional[str]:
    """
    Send a prompt to the judge model and return cleaned JSON output.

    Args:
        prompt: Judge prompt.

    Returns:
        Cleaned JSON response, or None on error.
    """
    try:
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        model_response = message.content[0].text.strip()
        clean_response = clean_json(model_response)
        logger.debug(f"Judge response: {clean_response}")
        return clean_response
    except Exception as e:
        logger.error(f"Judge error: {e}")
        return None
