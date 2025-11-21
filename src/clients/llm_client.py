import httpx

from src.core.config import MODEL_NAME, OLLAMA_API_URL

timeout: httpx.Timeout = httpx.Timeout(200.0)
client: httpx.AsyncClient = httpx.AsyncClient(timeout=timeout)


async def call_llm(prompt: str) -> str:
    """
    Sends a prompt to the configured LLM model and returns its response.

    Builds a JSON payload and sends it to the Ollama API, then extracts
    and returns the model's textual output.

    Args:
        prompt (str): Input text prompt for the language model.

    Returns:
        str: The generated response text (may be empty if none returned).

    Raises:
        httpx.HTTPStatusError: If the Ollama API returns a non-2xx status.
    """
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    response = await client.post(OLLAMA_API_URL, json=payload)
    response.raise_for_status()
    result = response.json()
    return result.get("response", "").strip()
