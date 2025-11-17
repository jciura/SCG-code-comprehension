import httpx
from mcp.server.fastmcp import FastMCP
from loguru import logger

mcp = FastMCP("junie-context")


@mcp.tool()
async def ask_junie(question: str) -> str:
    """
        Sends a question to the Junie RAG API and returns its contextual response.

        Forwards the question to the configured Junie service URL and extracts the
        `context` field from the JSON response. Returns any exception message as
        text if the request fails.

        Args:
            question (str): The user question to send to the Junie backend.

        Returns:
            str: The retrieved context string or an error message if the call fails.
    """
    logger.info("GOT QUESTION: {}", question)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "http://127.0.0.1:8000/ask_junie",
                json={"question": question},
            )
            response.raise_for_status()
            data = response.json()
            context = data.get("context", "No context found")
            return context
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    mcp.run()
