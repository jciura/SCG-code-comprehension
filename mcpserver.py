import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("junie-context")


@mcp.tool()
async def ask_junie(question: str) -> str:
    print("GOT QUESTION: " + question)
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
