from loguru import logger
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()


def judge_answer(prompt: str) -> str:

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}])
        model_response = message.content[0].text.strip()
        logger.debug(f"Judge response: {model_response}")
        return model_response
    except Exception as e:
        logger.error(f"Judge error: {e}")
        return None