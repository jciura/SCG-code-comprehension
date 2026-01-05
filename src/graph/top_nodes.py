import json
import re
import time
from typing import Any, Dict

from loguru import logger

from context.context_builder import build_context
from core.models import IntentAnalysis
from graph.NeighborTypeEnum import NeighborTypeEnum
from graph.QueryTopMode import QueryTopMode


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
        related_entities = node.get("related_entities", [])
        related_entities = json.loads(related_entities)
        return len(related_entities)
    else:
        return float(node.get(metric, 0.0))


def extract_json(text: str) -> dict:
    """
    Extract and parse a JSON object from text.
    """
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found")
    return json.loads(match.group())


async def get_top_nodes_context(
    question: str, analysis: IntentAnalysis, model_name: str, collection: Any, **params
) -> tuple[list[dict[str, float | Any]], str]:
    """
    Finds top nodes based on LLM-guided kind/metric selection.

    Args:
        question: Natural-language question
        collection: Chroma collection handle

    Returns:
        Top nodes with metadata and metric values
    """
    logger.info(f"Params: {params}")
    query_mode = params.get("query_mode", QueryTopMode.LIST_ONLY)
    query_mode = QueryTopMode(query_mode.lower())
    kinds = params.get("kinds", "ANY")
    if isinstance(kinds, str):
        kinds = [kinds]
    kinds = [NeighborTypeEnum[kind.upper()] for kind in kinds]
    metric = params.get("metric", "combined")
    limit = params.get("limit", 10)
    if isinstance(limit, str) and limit != "all":
        limit = int(limit)
    exact_metric_value = params.get("exact_metric_value", 0)
    order = params.get("order", "desc")
    if isinstance(query_mode, str):
        try:
            query_mode = QueryTopMode(query_mode)
        except ValueError:
            logger.warning(f"Invalid query_mode '{query_mode}', using LIST_ONLY.")
            query_mode = QueryTopMode.LIST_ONLY

    logger.info(f"Params: {query_mode, kinds, metric, limit, exact_metric_value, order}")
    start_time = time.time()

    where = None

    if NeighborTypeEnum.ANY not in kinds:
        where = {"kind": {"$in": [k.value for k in kinds]}}

    results = collection.get(include=["metadatas", "documents"], where=where)

    nodes = [
        {
            "node": results["ids"][i],
            "metadata": results["metadatas"][i],
            "code": results["documents"][i],
            "metric_value": get_metric_value(results["metadatas"][i], metric),
        }
        for i in range(len(results["ids"]))
    ]
    if limit == "all":
        top_nodes = [node for node in nodes if node["metric_value"] == exact_metric_value]
    else:
        top_nodes = sorted(
            (
                node
                for node in nodes
                if NeighborTypeEnum[node["metadata"].get("kind").upper()] in kinds
                or NeighborTypeEnum.ANY in kinds
            ),
            key=lambda n: n["metric_value"],
            reverse=(order.lower() == "desc"),
        )[:limit]

    logger.debug(f"MODE: {query_mode}")
    if query_mode == QueryTopMode.LIST_ONLY:
        context = " ".join(
            f"{node.get('metadata', {}).get('label', '')} - {
                node.get('metadata', '').get('kind', '')
            } - {node.get('metadata', '').get('uri', '')} - Metric value: {
                node.get('metric_value'):.2f}"
            for node in top_nodes
        )
    else:
        top_nodes = [
            (
                n["metric_value"],
                {
                    "node": n["node"],
                    "metadata": n["metadata"],
                    "code": n["code"],
                },
            )
            for n in top_nodes
        ]
        context = build_context(
            top_nodes, "definition", 1.0, len(top_nodes), question=question, target=None
        )

    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000
    logger.debug(f"Completed in: {elapsed_ms:.1f}ms")
    return top_nodes, context or "<NO CONTEXT FOUND>"
