import json
from typing import Any, Dict, List

from loguru import logger

from src.clients.llm_client import call_llm


def get_metric_value(node: Dict[str, Any], metric: str) -> float:
    """
    Returns a numeric metric value for a node.

    Args:
        node: Node metadata dictionary
        metric: Metric key (e.g., "combined", "pagerank", "in-degree", "out-degree",
            "number_of_neighbors")

    Returns:
        Metric value for the node
    """
    if metric == "number_of_neighbors":
        related_entities_str = node.get("related_entities", "")
        try:
            related_entities = (
                json.loads(related_entities_str)
                if isinstance(related_entities_str, str)
                else related_entities_str
            )
        except (json.JSONDecodeError, TypeError):
            logger.debug("Failed to parse related_entities, using empty list.")
            related_entities = []
        return len(related_entities)
    else:
        return float(node.get(metric, 0.0))


async def find_top_nodes(question: str, collection: Any) -> List[Dict[str, Any]]:
    """
    Finds top nodes based on LLM-guided kind/metric selection.

    Args:
        question: Natural-language question
        collection: Chroma collection handle

    Returns:
        Top nodes with metadata and metric values
    """
    classification_prompt = f"""
    User question: "{question}"
    
    Your task:
    1. Determine which node types (CLASS, METHOD, VARIABLE, PARAMETER, CONSTRUCTOR) are relevant
    2. Choose a ranking metric from: loc (lines of code), pagerank, eigenvector, in_degree, 
        out_degree, combined, number_of_neighbors
    3. Determine how many nodes the user wants
    4. Decide sort order (ascending "asc" or descending "desc"):
       - If question contains words like "biggest", "largest", "most", "max" → use "desc"
       - If question contains words like "smallest", "least", "min" → use "asc"
    5. For general questions about "most important", "key", "main", "core" elements → 
        choose "CLASS" and "combined"
    
    Return ONLY valid JSON format:
    {{"kinds": ["CLASS", "METHOD"], "metric": "combined", "limit": 5, "order": "desc"}}
    
    No comments, only JSON.
"""

    analysis = await call_llm(classification_prompt)
    logger.debug(f"Top nodes analysis: {analysis}")

    try:
        parsed = json.loads(analysis)
        kinds = parsed.get("kinds", ["CLASS"])
        metric = parsed.get("metric", "combined")
        order = parsed.get("order", "desc")
        limit = parsed.get("limit", 5)
    except json.JSONDecodeError:
        kinds = ["CLASS"]
        metric = "combined"
        order = "desc"
        limit = 5

    results = collection.get(include=["metadatas"])

    nodes = [
        {
            "node": results["ids"][i],
            "metadata": results["metadatas"][i],
            "metric_value": get_metric_value(results["metadatas"][i], metric),
        }
        for i in range(len(results["ids"]))
    ]
    logger.debug(f"Sample node: {nodes[0] if nodes else 'None'}")
    filtered_sorted_nodes = sorted(
        (node for node in nodes if node["metadata"].get("kind") in kinds),
        key=lambda n: n["metric_value"],
        reverse=(order.lower() == "desc"),
    )

    return filtered_sorted_nodes[:limit]
