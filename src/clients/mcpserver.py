from pathlib import Path
from typing import List, Tuple

import httpx
from loguru import logger
from mcp.server.fastmcp import FastMCP

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

@mcp.tool()
async def read_agent_instructions() -> str:
    """
    📖 READ THIS FIRST! Complete guide for SCG code analysis tools.

    Returns comprehensive instructions including:
    - Decision tree: which function to use for each question type
    - Parameter selection guidelines
    - Query examples with explanations
    - Checklist before submitting questions

    ⚠️ CRITICAL: Call this if you're unsure which tool to use or how to set parameters.

    Returns:
        str: Complete agent instructions with examples and decision flowchart
    """
    return f"""# SCG CODE ANALYSIS AGENT - INSTRUCTIONS {AGENT_INSTRUCTIONS} """

@mcp.tool()
async def ask_specific_nodes(
        question: str, pairs: List[Tuple[str, str]], top_k: int, max_neighbors: int, neighbor_types: List[str],
        relation_types: List[str]) -> str:
    """
    #REMINDER: IF YOU HAVE CONTEXT (CODE) OF NODE FROM ANOTHER QUERY AND YOU CAN GIVE ANSWER DO NOT ASK ANOTHER QUESTION.

    **When to use:**
    Question contains specific names of code elements (classes, methods, functions, variables, constructors).
    - Question contains a proper name: "LoginController", "authenticate", "userRepository" and type: CLASS,METHOD etc. If
      type is missing you can try to guess it based on
      whole conversation and name of node
    - Question asks: "Describe class X", "What does method Y do?", "What fields does class Z have?"
    - You want to find a specific element by name

    **HOW BACKEND WORKS:**
    - Inside server node is retrieved from vector base by string of `KIND NAME` and that is achieved by parameter `pairs`
    - User asks about specific **variable**, **parameter** or **value** ask him to specify which **METHOD/CLASS/INTERFACE** it belongs to. Data about parameters, variables and values is not stored in vector base.

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
      "pairs": [["kind1", "name1"], ["kind2", "name2"]],
      "top_k": 3-5,
      "max_neighbors": 1-8,
      "neighbor_types": ["CLASS|METHOD|CONSTRUCTOR|INTERFACE|ENUM|OBJECT|TYPE|TYPE_PARAMETER|ANY"]
    }
    ```

    **Parameters values selection:**


    - `pairs:`
      - Extract from user's question pairs (kind, name) for each entity mentioned in it. If there are mistakes in spelling - fix it.
      - Example: "Describe User Class" -> pairs = [["CLASS", "User"]]
      - Example: "Describe User class and Category class" -> pairs = [["CLASS", "User"], ["CLASS", "Category"]]

    - `max_neighbors:`
      - Simple question ("Describe class X") → **1-2**
      - Medium question ("Where is class X used?") → **3-5**
      - Complex question ("What are all dependencies of class X?") → **6-8**

    - `neighbor_types:` - specifies the list of **TYPES OF NEIGHBOR NODES** to fetch based on user question.

        Available options are: CLASS,METHOD,CONSTRUCTOR,INTERFACE,ENUM,OBJECT,TYPE,TYPE_PARAMETER,ANY

        - Question: "Describe User class" - `neighbor_types` not specified in question so go with ["ANY"]
        - Question: "Describe User class and 2 most important classes related to it" - `neighbor_types` is specified and it
          is ["CLASS"]
        - Question: "Where is class X used?" - `neighor_type` not specified - go with ["ANY"]
        - Question: "Desctibe User class and most imporatant methods and classes conntected to it" -> set `neighbor_type`
          to ["CLASS", "METHOD"]
        - Unsure what to choose - choose ["ANY"]

    **MISTAKES**:
    Using name of question node as `neighbor_types`

    ```json
    {
      "neighbor_types": "CategoryController",
      "question": "Describe CategoryController class"
    }
    ```
    ---
    """
    logger.info("MCP specific_nodes question: {}".format(question))
    params = {"top_k": top_k, "pairs": pairs, "max_neighbors": max_neighbors, "neighbor_types": neighbor_types,
              "relation_types": relation_types}
    return await call_fastapi("ask_specific_nodes", question, params)


@mcp.tool()
async def ask_top_nodes(question: str, query_mode: str, kinds: List[str], metric: str, limit: str,
                        exact_metric_value: int, order: str) -> str:
    """
    Query for RANKINGS, TOP-N elements, largest/smallest values in code.

    **When to use:**
    Question is about ranking, top-N elements, largest/smalles values.
    - Question contains: "top", "largest", "smallest", etc.
    - User asks about ordered list

    **HOW BACKEND WORKS:**
    - For `list_only` returns only list of selected nodes with `kind`, `uri` and `metric`.
    - For `full_desc` return code of selected nodes.

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
      "query_mode": "list_only|full_desc",
      "kinds": "[List of kinds]",
      "metric": "metric to filter nodes",
      "limit": "number of nodes to fetch",
      "exact_metric_value": "exact value of metric if present in question",
      "order": "desc|asc"
    }
    ```

    **Parameters values selection:**


    - `query_mode:`

        - `"list_only"` — ranking only without detailed descriptions
        - `"full_desc"` — ranking with full description of each element

        **Do not user `null`, empty string or other values**. Always choose one of two available modes.

    - `kinds:`

        `kinds` specifiecs the list of **TYPES OF NODES** to fetch based on user question.

        Available options are: CLASS,METHOD,VARIABLE,CONSTRUCTOR,ANY.

        - **HOW TO CHOOSE**:
          - Question: "What are 5 most important classes" - `kinds` specified in question so go with ["CLASS"]
          - Question: "What are all entities with none neighbors" - `kinds` is not specified so go with ["ANY"]
          - Question: "What are 5 most important classes or methods?" - `kinds` is specified - go with ["CLASS", "METHOD"]
          - Unsure what to choose - choose ["ANY"]


    - `metric:`
        `metric` specifiecs the metric user wants to filter nodes with.

        Available metrics are:
            - loc - (lines of code),
            - pagerank - importance based on the quantity and quality of links pointing to them
            - katz - measures a node's influence in a network by summing its direct and indirect connections, assigning a diminishing weight to longer paths, meaning immediate neighbors matter more than distant ones.
            - eigenvector - connection to most important entities
            - in_degree - number of ingoing edges
            - out_degree - number of outgoing edges
            - combined - combined metric, most important entities
            - number_of_neighbors - number of related entities'

         - **HOW TO CHOOSE**:
          - Question: "What are 5 most important classes" - `metric` is not specified in question so go with "combined"
          - Question: "What are all entities with none neighbors" - `metric` is specified so go with "number_of_neighbors"
          - Question: "What are 5 classes with most lines of code?" - `metric` is specified - go with "combined"
          - Unsure what to choose - choose "combined"

    - `limit:`
        `limit` specifies how many nodes to fetch based on user question.

        - **HOW TO CHOOSE**:
            - "all", "everything", "wszystkie" or something like that is in question -> "limit" = "all"
            - If the question contains a number connected to number of nodes -> "limit" = that number

    - `exact_metric_value:`
        `exact_metric_value specifies value of node metrics that needs to be fetched.

        - **HOW TO CHOOSE**:
            - If limit = "all" AND user explicitly mentions a metric value (example: "with none neighbors", "with 0 neighbors", "with 0 lines of code") → metric_value = that value
                -Treat words like "none", "no", "without", "brak" as 0
            - Otherwise → metric_value = 0

    - `order:`
        `order` specifies order in which list of nodes is sorted

        - **HOW TO CHOOSE**:
           - If question contains words like "biggest", "largest", "most", "max" → use "desc"
           - If question contains words like "smallest", "least", "min" → use "asc"
           - If not sure → order = "desc"

    **Call examples:**
    ```json
    {
      "question": "What are 5 most important classes",
      "query_mode": "list_only"
      "kinds": ["CLASS"],
      "metric": "combined"
      "limit": 5,
      "exact_metric_value": 0,
      "order": "desc"
    }
    ```

    ```json
    {
      "question": "Describe 5 most important classes",
      "query_mode": "full_desc",
      "kinds": ["CLASS"],
      "metric": "combined",
      "limit": 5,
      "exact_metric_value": 0,
      "order": "desc"
    }
    ```

    ```json
    {
      "question": "What are classes without neighbors",
      "query_mode": "list_only",
      "kinds": ["CLASS"],
      "metric": "number_of_neighbors",
      "limit": "all",
      "exact_metric_value": 0,
      "order": "desc"
    }
    ```
    ---
    DON'T BE STUCK ON THIS QUESTION, just list nodes that you get with metric that you get for list_only.
    """
    logger.info("MCP top_nodes question: {}".format(question))
    params = {"query_mode": query_mode, "kinds": kinds, "metric": metric, "limit": limit,
              "exact_metric_value": exact_metric_value, "order": order}
    return await call_fastapi("ask_top_nodes", question, params)


@mcp.tool()
async def ask_general_question(question: str, kinds: List[str], keywords: List[str], top_nodes: int,
                               max_neighbors: int) -> str:
    """
    #REMINDER: IF YOU HAVE CONTEXT (CODE) OF NODE FROM ANOTHER QUERY AND YOU CAN GIVE ANSWER DO NOT ASK ANOTHER QUESTION.

    Query for GENERAL/ARCHITECTURAL information about code without specific targets.

   **When to use:**
    Question is about architecture, logic flow, general system behavior. No specific nodes names are mentioned in question.
    - No specific element names in the question
    - Question about patterns, concepts, flows

    **HOW BACKEND WORKS:**
    - Based on `kinds` and `keywords` chooses nodes that can potentially answer to question. Candidates to choose are later evaluated by other LLM using snippet of their code.
      You need to guess that `kinds` and `keywords`.

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
      "kinds": ["List of kinds that can backend should search for"],
      "keywords": ["List of keywords that names of code entities can include"],
      "top_nodes": 5-8,
      "max_neighbors": 2-5
    }
    ```

    **Parameters values selection:**

    - `kinds` - based on question choose kinds of nodes that can be related to question
      - possible values: **CLASS**, **METHOD**, **INTERFACE**, **ENUM**, **TYPE_PARAMETER**, **OBJECT**, **TYPE**
    - `keywords` - based on question choose 10 keywords that should be included in nodes names in Java or Scala
      - example: Question: "How is logging implemented" --> some keywords: ["login", "controller", "auth", "authenticate"]
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
      "max_neighbors": 3,
      "kinds": ["CLASS", "METHOD"],
      "keywords": ["login", "controller", "view", "auth", ...]
    }
    ```

    Complex Question

    ```json
    {
      "question": "Describe the entire authentication system architecture",
      "top_nodes": 8,
      "max_neighbors": 5,
      "kinds": ["CLASS", "METHOD"],
      "keywords": ["auth", ...]
    }
    ```
    ---

    #WORKFLOW TIP:
    1. Start with ask_general_question to get overview
    2. Extract specific element names from results
    3. Suggest using ask_specific_nodes for detailed analysis of interesting elements to user
    """
    logger.info("MCP general_question question: {}".format(question))
    params = {"top_nodes": top_nodes, "kinds": kinds, "keywords": keywords, "max_neighbors": max_neighbors}
    return await call_fastapi("ask_general_question", question, params)


@mcp.tool()
async def list_related_entities(question: str, pairs: List[Tuple[str, str]], limit: int, neighbor_types: List[str],
                                relation_types: List[str]) -> str:
    """
    #REMINDER: IF YOU HAVE CONTEXT (CODE) OF NODE FROM ANOTHER QUERY AND YOU CAN GIVE ANSWER DO NOT ASK ANOTHER QUESTION.

    Query to get all classed/methods etc. connected to node in question.

    **When to use:**
    Query to get all classed/methods etc. connected to node in question.

    - There is a specific node type and name in user's question.
    - User wants to list entities connected to node in question.


    **HOW BACKEND WORKS:**
    - Based on parameters that you select backend query nodes that match criteria. It return list of `node_id` - `kind` - `uri` - `type or relation to node in question`



    **Examples:**

    - "What are neighbors of class X"
    - "To what classes is class X connected"

    **Parameters:**
    ```json
    {
      "question": "exact user question"(can be changed minimally),
      "pairs": [["kind1", "name1"]],
      "limit": number/"all",
      "neighbor_types": ["CLASS|METHOD||CONSTRUCTOR|INTERFACE|ENUM|TYPE_PARAMETER|ANY"],
      "relation_types": ["DECLARATION|DECLARATION_BY|CALL|CALL_BY|RETURN_TYPE_ARGUMENT|RETURN_TYPE_ARGUMENT_BY|ANY"]
    }
    ```
    **Parameters values selection:**

    - `pairs:`
      - Extract from user's question pairs (kind, name) for each entity mentioned in it. If there are mistakes in spelling - fix it.
      - Example: "What are 5 most important methods called by class X?" -> pairs = [["CLASS", "X"]]
      - Example: "What are all classes method Y is called by?" -> pairs = [["METHOD", "Y"]]
      - If there is more pairs than one go with two query, one for each pair to don't get too much data.

    - `limit` - represents how many neighbors related to node user wants to get.
      - **HOW TO CHOOSE**:
        - Question: "What are neighbors of class X" -> not specified so go with "all"
        - Question: "To what classes is class X connected" -> not specified so go with "all"
        - Question: "What are 5 most important classes connected to method X" -> specified so go with 5

    - `neighbor_types` - specifies the list of **TYPES OF NEIGHBOR NODES** to fetch based on user question.
      Available options are: CLASS,METHOD,VARIABLE,CONSTRUCTOR,ANY.
      - **HOW TO CHOOSE**:
        - Question: "What are 5 most important classes connected to method X?"" - `neighbor_types` is specified in question so it is ["CLASS"]
        - Question: "What are neighbors of class X?" - `neighbor_types` is not specified so go with ["ANY"]
        - Question: "What are 5 most important classes or methods connected to method X?" - `neighbor_type` is specified - it is ["CLASS", "METHOD"]
        - Unsure what to choose - choose ["ANY"]

    - `relation_types` - specifies the relation types to fetch based on user question.
      Available options are: DECLARATION,DECLARATION_BY,CALL,CALL_BY,RETURN_TYPE_ARGUMENT,RETURN_TYPE_ARGUMENT_BY,ANY.
      - **HOW TO CHOOSE**:
        - Question: "What are 5 most important methods called by class X?"" - `relation_types` is specified -> ["CALL"]
        - Question: "What are all classes method Y is called by?" - `relation_types` is specified -> ["CALL_BY"]
        - Question: "What are all neighbors of class X?" - `relation_types` is not specified -> ["ANY"]
        - Unsure what to choose - choose ["ANY"]
    """
    logger.info("MCP list related_entities question: {}".format(question))
    params = {"pairs": pairs, "limit": limit, "neighbor_types": neighbor_types, "relation_types": relation_types}
    return await call_fastapi("list_related_entities", question, params)

if __name__ == "__main__":
    try:
        mcp.run()
    except Exception:
        logger.exception("MCP server failed")
