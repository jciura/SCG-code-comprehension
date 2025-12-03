from pathlib import Path
from typing import List

import httpx
from loguru import logger
from mcp.server.fastmcp import FastMCP
from graph.QueryTopMode import QueryTopMode

mcp = FastMCP("scg-context")

async def call_fastapi(endpoint: str, question: str, params) -> str:
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
                json={"question": question, "params": params},
            )
            response.raise_for_status()
            data = response.json()
            prompt = data.get("prompt", "No context found")
            return prompt
    except Exception as e:
        return str(e)


AGENT_INSTRUCTIONS = ""
try:
    agent_md_path = Path(__file__).parent / "AGENT.md"
    with open(agent_md_path, "r", encoding="utf-8") as f:
        AGENT_INSTRUCTIONS = f.read()
except Exception as e:
    logger.warning(f"Could not load AGENT.md: {e}")


@mcp.resource("file://agent-instructions")
async def get_agent_instructions() -> str:
    """
    SCG Agent Instructions - Critical guidelines for tool usage.
    READ THIS BEFORE USING ANY TOOLS.
    """
    return AGENT_INSTRUCTIONS


@mcp.tool()
async def read_agent_instructions() -> str:
    """
    📖 READ THIS FIRST! Complete guide for SCG code analysis tools.

    Returns comprehensive instructions including:
    - Decision tree: which function to use for each question type
    - Parameter selection guidelines (max_neighbors, top_k, top_nodes)
    - neighbor_types usage rules and common mistakes
    - Query examples with explanations
    - Checklist before submitting questions

    ⚠️ CRITICAL: Call this if you're unsure which tool to use or how to set parameters.

    Returns:
        str: Complete agent instructions with examples and decision flowchart
    """
    return f"""# SCG CODE ANALYSIS AGENT - INSTRUCTIONS {AGENT_INSTRUCTIONS} """

@mcp.tool()
async def ask_specific_nodes(
        question: str, top_k: int, max_neighbors: int, neighbor_types: List[str]) -> str:
    """
    #REMINDER: IF YOU HAVE CONTEXT (CODE) OF NODE FROM ANOTHER QUERY AND YOU CAN GIVE ANSWER DO NOT ASK ANOTHER QUESTION.

    Query for specific code elements (classes, methods, functions, variables, constructors).

    **Use when:**
    - Question contains a proper name: "LoginController", "authenticate", "userRepository" and type: CLASS,METHOD etc. If type is missing you can try to guess it based on
     whole conversation and name of node
    - Question asks: "Describe class X", "What does method Y do?", "What fields does class Z have?"
    - You want to find a specific element by name

    **DON'T use when:**
    - Question is general: "How does login work?", "Describe the architecture", "How is logging implemented?"
    - No specific code element names present
    - Asking about "top X" or ranking

    **Examples:**

    - "How is LoginController class implemented?"
    - "What does authenticate method in AuthService do?"
    - "Desctibe the User class"

    **Parameters:**
    ```json
    {
      "question": "exact user question",
      "top_k": 3-5,
      "max_neighbors": 1-8,
      "neighbor_types": ["CLASS|METHOD|VARIABLE|CONSTRUCTOR|ANY"]
    }
    ```

    **Choosing max_neighbors:**

    - Simple question ("Describe class X") → **1-2**
    - Medium question ("Where is class X used?") → **3-5**
    - Complex question ("What are all dependencies of class X?") → **6-8**

    **neighbor_types:**
    `neighbor_types` specifiecs the list of **TYPES OF NEIGHBOR NODES** to fetch based on user question.

    Available options are: CLASS,METHOD,VARIABLE,CONSTRUCTOR,ANY.

    - **HOW TO CHOOSE**:
      - Question: "Describe User class" - `neighbor_types` not specified in question so go with ["ANY"]
      - Question: "Describe User class and 2 most important classes related to it" - `neighbor_types` is specified and it is ["CLASS"]
      - Question: "Where is class X used?" - `neighor_type` not specified - go with ["ANY"]
      - Question: "Desctibe User class and most imporatant methods and classes conntected to it" -> set `neighbor_type` to ["CLASS", "METHOD"]
      - Unsure what to choose - choose ["ANY"]

    **MISTAKES**:
    Using name of question node as `neighbor_types`
    ```json
    {
      "neighbor_types": "CategoryController",
      "question": "Describe CategoryController class"
    }


    CHECKLIST BEFORE CALLING:
    ☐ Question passed exactly as user asked? (minimal changes only)
    ☐ Question contains specific element names? (not a general question)
    ☐ max_neighbors matches question complexity? (1-2 simple, 3-5 medium, 6-8 complex)
    ☐ neighbor_types are TYPES (CLASS/METHOD/etc.), not element names?
    ☐ neighbor_types match what user is asking about?
    """
    logger.info("MCP specific_nodes question: {}".format(question))
    params = {"top_k": top_k, "max_neighbors": max_neighbors, "neighbor_types": neighbor_types}
    return await call_fastapi("ask_specific_nodes", question, params)


@mcp.tool()
async def ask_top_nodes(question: str, query_mode: QueryTopMode) -> str:
    """
    Query for RANKINGS, TOP-N elements, largest/smallest values in code.

    **Use when:**
    - Question contains: "top", "largest", "smallest", etc.
    - User asks about ordered list

    **DON'T use when:**
    - Question about a specific element name
    - General question about system behavior

    **Examples:**

    - "What are 5 classes with the most lines of code?"
    - "Top 3 functions with the most neighbors"
    - "Describe 5 most important classes"
    - "Describe 5 most important elements"

    **Parameters:**
    ```json
    {
      "question": "exact user question",
      "query_mode": "list_only|full_desc"
    }
    ```

    **query_mode:**

    - `"list_only"` — ranking only without detailed descriptions
    - `"full_desc"` — ranking with full description of each element

    **Do not user `null`, empty string or other values**. Always choose one of two available modes.

    **Call examples:**
    ```json
    {
      "question": "What are 5 most imporatant classes",
      "query_mode": "list_only"
    }
    ```

    ```json
    {
      "question": "Describe 5 most imporatant classes",
      "query_mode": "full_desc"
    }
    ```

     CHECKLIST BEFORE CALLING:
    ☐ Question passed exactly as user asked? (minimal changes only)
    ☐ Question asks for ranking/top-N? (not a specific element)
    ☐ query_mode is exactly "list_only" or "full_desc"? (no other values)
    ☐ query_mode matches user's intent? (list vs detailed description)
    """
    logger.info("MCP top_nodes question: {}".format(question))
    params = {"query_mode": query_mode.value}
    return await call_fastapi("ask_top_nodes", question, params)


@mcp.tool()
async def ask_general_question(question: str, top_nodes: int, max_neighbors: int) -> str:
    """
    #REMINDER: IF YOU HAVE CONTEXT (CODE) OF NODE FROM ANOTHER QUERY AND YOU CAN GIVE ANSWER DO NOT ASK ANOTHER QUESTION.

    Query for GENERAL/ARCHITECTURAL information about code without specific targets.

    **Use when:**"
    - No specific element names in the question
    - Question about patterns, concepts, flows

    **DON'T use when:**
    - Question contains a specific class/method name
    - Question about ranking/top X

    **Examples:**

    - "Describe the user login implementation"
    - "How does the authentication module work?"

    **Parameters:**
    ```json
    {
      "question": "exact user question",
      "top_nodes": 5-8,
      "max_neighbors": 2-5
    }
    ```

    **Parameters values selection:**

    - `top_nodes` — how many main nodes to select for analysis:
      - Simple question: 5-6
      - Complex question: 7-8
    - `max_neighbors` — how many neighbors to fetch for every main node
      - Simple question: 2-3
      - Complex question: 4-5


    **Selection examples:**

    Simple question
    ```json
    {
      "question": "How does login work?",
      "top_nodes": 5,
      "max_neighbors": 3
    }
    ```

    Complex Question
    ```json
    {
      "question": "Describe the entire authentication system architecture",
      "top_nodes": 8,
      "max_neighbors": 5
    }
    ```

    #WORKFLOW TIP:
    1. Start with ask_general_question to get overview
    2. Extract specific element names from results
    3. Suggest using ask_specific_nodes for detailed analysis of interesting elements to user

    #CHECKLIST BEFORE CALLING:
    ☐ Question passed exactly as user asked? (minimal changes only)
    ☐ Question is general/architectural? (no specific element names)
    ☐ top_nodes matches complexity? (5-6 simple, 7-8 complex)
    ☐ max_neighbors matches depth needed? (2-3 simple, 4-5 complex)
    ☐ Total context (top_nodes × max_neighbors) reasonable?
    """
    logger.info("MCP general_question question: {}".format(question))
    params = {"top_nodes": top_nodes, "max_neighbors": max_neighbors}
    return await call_fastapi("ask_general_question", question, params)


if __name__ == "__main__":
    try:
        mcp.run()
    except Exception:
        logger.exception("MCP server failed")
