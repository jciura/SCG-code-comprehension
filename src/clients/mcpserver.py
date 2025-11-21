import httpx
from loguru import logger
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("junie-context")


async def call_fastapi(endpoint: str, question: str) -> str:
    """
    Sends a question to the Junie RAG API and returns its contextual response.

    Forwards the question to the configured Junie service URL and extracts the
    `context` field from the JSON response. Returns any exception message as
    text if the request fails.

    Args:
        endpoint: 
        question (str): The user question to send to the Junie backend.

    Returns:
        str: The retrieved context string or an error message if the call fails.
    """
    logger.info("GOT QUESTION: {}", question)
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(
                f"http://127.0.0.1:8000/{endpoint}",
                json={"question": question},
            )
            response.raise_for_status()
            data = response.json()
            prompt = data.get("prompt", "No context found")
            return prompt
    except Exception as e:
        return str(e)


@mcp.tool()
async def ask_specific_nodes(question: str) -> str:
    """
    Pytanie, w którym wiadomo jaki jest typ i nazwa węzła lub węzłów jakich mamy szukać.
    """
    logger.info("MCP specific_nodes question: {}".format(question))
    return await call_fastapi("ask_specific_nodes", question)


@mcp.tool()
async def ask_top_nodes(question: str) -> str:
    """
    Pytanie typu top - szukamy węzłow o najmniejzej/największej wartości parametru.
    """
    logger.info("MCP top_nodes question: {}".format(question))
    return await call_fastapi("ask_top_nodes", question)


@mcp.tool()
async def ask_general_question(question: str) -> str:
    """
    Ogólne pytanie, w którym nie są znane szukane węzły lub nawet jeżeli jakiś jest podany
    to i tak jest ono bardzo ogólne.
    """
    logger.info("MCP general_question question: {}".format(question))
    return await call_fastapi("ask_general_question", question)


if __name__ == "__main__":
    try:
        mcp.run()
    except Exception:
        logger.exception("MCP server failed")
