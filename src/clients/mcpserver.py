import sys
from pathlib import Path
from typing import List

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
project_root = Path(__file__).resolve().parent.parent.parent
logger.remove()
log_file = project_root / "server_debug.log"
logger.add(log_file, level="DEBUG")

try:
    from mcp.server.fastmcp import FastMCP
    from app.main import retrieve_context
    from graph.general_query import get_general_nodes_context
    from graph.related_entities import get_related_entities
    from graph.specific_nodes import get_specific_nodes_context
    from graph.top_nodes import get_top_nodes_context
except ImportError as e:
    logger.error(f"Failed to import FastMCP module: {e}")
    sys.exit(1)

mcp = FastMCP("scg-context")


async def call_backend(function, question: str, params) -> str:
    """
    Sends a question to the Junie RAG API and returns its contextual response.

    Forwards the question to the configured Junie service URL and extracts the
    `context` field from the JSON response. Returns any exception message as
    text if the request fails.

    Args:
        question (str): The user question to send to the Junie backend.
        function: Function to retrieve context from.
        params (dict): Dict of parameters for function to retrieve context.

    Returns:
        str: The retrieved context string or an error message if the call fails.
    """
    logger.info("GOT QUESTION: {}", question)
    try:
        data = await retrieve_context(question, function, params)
        logger.debug(f"Received data: {data}")
        prompt = data.get("prompt", "No context found")
        return prompt
    except Exception as e:
        return str(e)


@mcp.tool()
async def ask_specific_nodes(
    question: str,
    pairs: List[List[str]],
    top_k: int,
    max_neighbors: int,
    neighbor_types: List[str],
    relation_types: List[str],
) -> str:
    """
    #REMINDER: IF YOU HAVE CONTEXT (CODE) OF NODE FROM ANOTHER QUERY
    AND YOU CAN GIVE ANSWER DO NOT ASK ANOTHER QUESTION.
    Retrieves code and context for specific classes, methods, interfaces, etc...,
    **If user asks about parameter or variable ask him to specify which
     method/class/interface it belongs to.**

    **When to use:**
    Question contains specific names of
    code elements (classes, methods, functions, variables, constructors).
    - Question contains a proper name: "LoginController", "authenticate",
     "userRepository" and type: CLASS,METHOD etc. If
      type is missing you can try to guess it based on
      whole conversation and name of node
    - Question asks: "Describe class X", "What does method Y do?", "What fields does class Z have?"
    - You want to find a specific element by name

    **HOW BACKEND WORKS:**
    - Inside server node is retrieved from vector base by string of
      `KIND NAME` and that is achieved by parameter `pairs`
    - User asks about specific **variable**, **parameter**
      or **value** ask him to specify which **METHOD/CLASS/INTERFACE** it belongs to.
      Data about parameters, variables and values is not stored in vector base.

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
      - Extract from user's question pairs (kind, name) for each entity mentioned in it.
        If there are mistakes in spelling - fix it.
      - Example: "Describe User Class" -> pairs = [["CLASS", "User"]]
      - Example: "Describe User class
        and Category class" -> pairs = [["CLASS", "User"], ["CLASS", "Category"]]

    - `max_neighbors:`
      - Simple question ("Describe class X") → **1-2**
      - Medium question ("Where is class X used?") → **3-5**
      - Complex question ("What are all dependencies of class X?") → **6-8**

    - `neighbor_types:` - specifies the list of **TYPES OF NEIGHBOR NODES** to fetch
        based on user question.

        Available options are:
        CLASS,METHOD,CONSTRUCTOR,INTERFACE,ENUM,OBJECT,TYPE,TYPE_PARAMETER,ANY

        - Question: "Describe User class" - `neighbor_types` not specified in
        question so go with ["ANY"]
        - Question: "Describe User class and 2 most important classes
            related to it" - `neighbor_types` is specified, and it is ["CLASS"]
        - Question: "Where is class X used?" - `neighbor_type` not specified - go with ["ANY"]
        - Question: "Describe User class and most important methods
            and classes connected to it" -> set `neighbor_type` to ["CLASS", "METHOD"]
        - Unsure what to choose - choose ["ANY"]

    ---
    ### 🚨 CRITICAL INSTRUCTION: STOP AND ANSWER 🚨

    **IF YOU RECEIVE CONTEXT FROM THIS TOOL, DO NOT CALL IT AGAIN FOR THE SAME NODE.**

    1. **You already have the answer:** The return value of this function
        contains the code you need.
    2. **Do not loop:** Do not try to "verify" or "fetch more" immediately.
    3. **Formulate the response:**
       - Describe the class/method logic in detail using the code provided.
       - List fields, explain methods, and handle the user's request.
    4. **Suggest, Don't Act:** - If you think more neighbors are needed,
        **ASK THE USER FIRST** in your text response.
       - *Example:* "I have analyzed `ContextWaiter`. It connects to `ReentrantLock`. Would you like
            me to fetch the code for `ReentrantLock` next?"
       - **DO NOT** trigger the tool for `ReentrantLock` automatically. Wait for user confirmation.
    """
    logger.info("MCP specific_nodes question: {}".format(question))
    params = {
        "top_k": top_k,
        "pairs": pairs,
        "max_neighbors": max_neighbors,
        "neighbor_types": neighbor_types,
        "relation_types": relation_types,
    }
    return await call_backend(get_specific_nodes_context, question, params)


@mcp.tool()
async def ask_top_nodes(
    question: str,
    query_mode: str,
    kinds: List[str],
    metric: str,
    limit: str,
    exact_metric_value: int,
    order: str,
) -> str:
    """
    Query for RANKINGS, TOP-N elements, largest/smallest values in code.

    **When to use:**
    Question is about ranking, top-N elements, largest/smallest values.
    - Question contains: "top", "largest", "smallest", etc.
    - User asks about ordered list

    **HOW BACKEND WORKS:**
    - For `list_only` returns only list of selected nodes with `kind`, `uri` and `metric`.
    - For `full_desc` return code of selected nodes.

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

        **Do not user `null`, empty string or other values**. Always choose one of two
            available modes.

    - `kinds:`

        `kinds` specifies the list of **TYPES OF NODES** to fetch based on user question.

        Available options are: CLASS,METHOD,VARIABLE,CONSTRUCTOR,ANY.

        - **HOW TO CHOOSE**:
          - Question: "What are 5 most important classes" - `kinds` specified in question
            so go with ["CLASS"]
          - Question: "What are all entities with none neighbors" - `kinds` is not specified
            so go with ["ANY"]
          - Question: "What are 5 most important classes or methods?" - `kinds`is specified -
            go with ["CLASS", "METHOD"]
          - Unsure what to choose - choose ["ANY"]


    - `metric:`
        `metric` specifies the metric user wants to filter nodes with.

        Available metrics are:
            - loc - (lines of code),
            - pagerank - importance based on the quantity and quality of links pointing to them
            - katz - measures a node's influence in a network by summing its direct
                and indirect connections, assigning a diminishing weight to longer paths,
                meaning immediate neighbors matter more than distant ones.
            - eigenvector - connection to most important entities
            - in_degree - number of ingoing edges
            - out_degree - number of outgoing edges
            - combined - combined metric, most important entities
            - number_of_neighbors - number of related entities'

         - **HOW TO CHOOSE**:
          - Question: "What are 5 most important classes" - `metric` is not specified in
            question so go with "combined"
          - Question: "What are all entities with none neighbors" - `metric` is specified
            so go with "number_of_neighbors"
          - Question: "What are 5 classes with most lines of code?" - `metric` is
            specified - go with "combined"
          - Unsure what to choose - choose "combined"

    - `limit:`
        `limit` specifies how many nodes to fetch based on user question.

        - **HOW TO CHOOSE**:
            - "all", "everything", "wszystkie" or something like
                that is in question -> "limit" = "all"
            - If the question contains a number connected to
                number of nodes -> "limit" = that number

    - `exact_metric_value:`
        `exact_metric_value specifies value of node metrics that needs to be fetched.

        - **HOW TO CHOOSE**:
            - If limit = "all" AND user explicitly mentions a metric value (example:
                "with none neighbors", "with 0 neighbors",
                "with 0 lines of code") → metric_value = that value
                -Treat words like "none", "no", "without", "brak" as 0
            - Otherwise → metric_value = 0

    - `order:`
        `order` specifies order in which list of nodes is sorted

        - **HOW TO CHOOSE**:
           - If question contains words like "biggest", "largest", "most", "max" → use "desc"
           - If question contains words like "smallest", "least", "min" → use "asc"
           - If not sure → order = "desc"


    ### 🚨 CRITICAL INSTRUCTION: STOP AND ANSWER 🚨

    **IF YOU RECEIVE CONTEXT FROM THIS TOOL, DO NOT CALL IT AGAIN FOR THE SAME NODE.**

    1. **Format List**: Present the results clearly (e.g., a numbered Markdown list).
    2. **Interpret**: Explain *why* these nodes are top-ranked based on the metric.
    3. **Suggest Next Step**: If you used `list_only`, the user cannot see the implementation.
        You MUST suggest analyzing the #1 result.
       *Example:* "These are the top 5 largest classes. Would you like me to analyze
        the implementation of the largest one, `MainController`?"
    """
    logger.info("MCP top_nodes question: {}".format(question))
    params = {
        "query_mode": query_mode,
        "kinds": kinds,
        "metric": metric,
        "limit": limit,
        "exact_metric_value": exact_metric_value,
        "order": order,
    }
    return await call_backend(get_top_nodes_context, question, params)


@mcp.tool()
async def ask_general_question(
        question: str, kinds: List[str], keywords: List[str], top_k: int, max_neighbors: int
) -> str:
    """
     #REMINDER: IF YOU HAVE CONTEXT (CODE) OF NODE FROM ANOTHER QUERY
         AND YOU CAN GIVE ANSWER DO NOT ASK ANOTHER QUESTION.

     Query for GENERAL/ARCHITECTURAL information about code without specific targets.

    **When to use:**
     Question is about architecture, logic flow, general system behavior.
     No specific nodes names are mentioned in question.
     - No specific element names in the question
     - Question about patterns, concepts, flows

     **HOW BACKEND WORKS:**
     - Based on `kinds` and `keywords` chooses nodes that can potentially answer to question.
      Candidates to choose are later evaluated by other LLM using snippet of their code.
       You need to guess that `kinds` and `keywords`.

     **Examples:**

     - "Describe the user login implementation"
     - "How does the authentication module work?"

     **Parameters:**
     ```json
     {
       "question": "exact user question",
       "kinds": ["List of kinds that can backend should search for"],
       "keywords": ["List of keywords that names of code entities can include"],
       "top_k": 5-8,
       "max_neighbors": 2-5
     }
     ```

     **Parameters values selection:**

     - `kinds` - based on question choose kinds of nodes that can be related to question
       - possible values:
        **CLASS**, **METHOD**, **INTERFACE**, **ENUM**, **TYPE_PARAMETER**, **OBJECT**, **TYPE**
     - `keywords` - based on question choose 10 keywords that should be included
         in nodes names in Java or Scala
       - example: Question: "How is logging implemented" --> some keywords:
         ["login", "controller", "auth", "authenticate"]
     - `top_k` — how many main nodes to select for analysis:
         - Simple question: 5-6
         - Complex question: 7-8
     - `max_neighbors` — how many neighbors to fetch for every main node
         - Simple question: 2-3
         - Complex question: 4-5


     ---
     ### 🚨 CRITICAL INSTRUCTION: STOP AND ANSWER 🚨

     **IF YOU RECEIVE CONTEXT FROM THIS TOOL, DO NOT CALL IT AGAIN FOR THE SAME NODE.**

     1. **Synthesize Architecture**: Combine the returned nodes to
         describe the flow or system design.
     2. **Highlight Gaps**: A general query is rarely perfect.
         Admit what parts of the flow seem to be missing.
     3. **Suggest Next Step**: Pick the most interesting specific class found
         in this search and suggest drilling down.
        *Example:* "It seems `SecurityConfig` is central to this flow.
        Would you like to see its detailed configuration?"
    """
    logger.info("MCP general_question question: {}".format(question))
    params = {
        "top_k": top_k,
        "kinds": kinds,
        "keywords": keywords,
        "max_neighbors": max_neighbors,
    }
    return await call_backend(get_general_nodes_context, question, params)


@mcp.tool()
async def list_related_entities(
    question: str,
    pairs: List[List[str]],
    limit: int,
    neighbor_types: List[str],
    relation_types: List[str],
) -> str:
    """
    #REMINDER: IF YOU HAVE CONTEXT (CODE) OF NODE FROM ANOTHER QUERY
        AND YOU CAN GIVE ANSWER DO NOT ASK ANOTHER QUESTION.

    Query to get all classed/methods etc. connected to node in question.

    **When to use:**
    Query to get all classed/methods etc. connected to node in question.

    - There is a specific node type and name in user's question.
    - User wants to list entities connected to node in question.


    **HOW BACKEND WORKS:**
    - Based on parameters that you select backend query nodes that match criteria.
        It returns list of `node_id` - `kind` - `uri` - `type or relation to node in question`


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
      "relation_types":
      ["DECLARATION|DECLARATION_BY|CALL|CALL_BY|RETURN_TYPE_ARGUMENT|RETURN_TYPE_ARGUMENT_BY|ANY"]
    }
    ```
    **Parameters values selection:**

    - `pairs:`
      - Extract from user's question pairs (kind, name) for each entity mentioned in it.
        If there are mistakes in spelling - fix it.
      - Example: "What are 5 most important methods called by class X?" -> pairs = [["CLASS", "X"]]
      - Example: "What are all classes method Y is called by?" -> pairs = [["METHOD", "Y"]]
      - If there is more pairs than one go with two query,
        one for each pair to don't get too much data.

    - `limit` - represents how many neighbors related to node user wants to get.
      - **HOW TO CHOOSE**:
        - Question: "What are neighbors of class X" -> not specified so go with "all"
        - Question: "To what classes is class X connected" -> not specified so go with "all"
        - Question: "What are 5 most important classes
            connected to method X" -> specified so go with 5

    - `neighbor_types` - specifies the list of **TYPES OF NEIGHBOR NODES**
        to fetch based on user question.
      Available options are: CLASS,METHOD,VARIABLE,CONSTRUCTOR,ANY.
      - **HOW TO CHOOSE**:
        - Question: "What are 5 most important classes connected to method X?"" - `neighbor_types`
            is specified in question so it is ["CLASS"]
        - Question: "What are neighbors of class X?" - `neighbor_types` is not specified
            so go with ["ANY"]
        - Question: "What are 5 most important classes or methods connected to method X?" -
            `neighbor_type` is specified - it is ["CLASS", "METHOD"]
        - Unsure what to choose - choose ["ANY"]

    - `relation_types` - specifies the relation types to fetch based on user question.
      Available options are:
      DECLARATION,DECLARATION_BY,CALL,CALL_BY,RETURN_TYPE_ARGUMENT,RETURN_TYPE_ARGUMENT_BY,ANY.
      - **HOW TO CHOOSE**:
        - Question: "What are 5 most important methods called by class X?"" - `relation_types`
            is specified -> ["CALL"]
        - Question: "What are all classes method Y is called by?" - `relation_types`
            is specified -> ["CALL_BY"]
        - Question: "What are all neighbors of class X?" - `relation_types`
            is not specified -> ["ANY"]
        - Unsure what to choose - choose ["ANY"]

    ### 🚨 CRITICAL INSTRUCTION: STOP AND ANSWER 🚨

    **IF YOU RECEIVE CONTEXT FROM THIS TOOL, DO NOT CALL IT AGAIN FOR THE SAME NODE.**

    1. **Summarize Relationships**: Group the connections
        (e.g., "Method X is called by these 3 controllers...").
    2. **Suggest Next Step**: Ask if the user wants to investigate the
        implementation of a specific related entity.
       *Example:* "Class X calls `DatabaseService` frequently. Would you like to inspect
       `DatabaseService` to see how it handles the query?"
    """
    logger.info("MCP list related_entities question: {}".format(question))
    params = {
        "pairs": pairs,
        "limit": limit,
        "neighbor_types": neighbor_types,
        "relation_types": relation_types,
    }
    return await call_backend(get_related_entities, question, params)


if __name__ == "__main__":
    try:
        mcp.run()
    except Exception as e:
        logger.exception("MCP server failed")
