import json
import re
import time
from typing import Any, Dict, List, Set, Tuple

from loguru import logger

from context.context_builder import build_context
from core.config import COMBINED_MAX
from core.models import IntentAnalysis, IntentCategory
from graph.retrieval_utils import is_child_of
from src.clients.llm_client import call_llm
from src.graph.usage_finder import find_usage_nodes

max_usage_nodes_for_context = 3


def _filter_candidates(
    all_nodes: Dict[str, Any], keywords: List[str], kind_weights: Dict[str, float]
) -> List[Tuple[str, Dict[str, Any], str, float]]:
    """
    Filters and scores candidate nodes based on LLM analysis.

    Args:
        all_nodes: All nodes from collection
        keywords: List of keywords to search for
        kind_weights: Weights for different node kinds

    Returns:
        List of candidate nodes with hybrid scores
    """
    candidate_nodes = []

    for i in range(len(all_nodes["ids"])):
        node_id = all_nodes["ids"][i]
        metadata = all_nodes["metadatas"][i]
        doc = all_nodes["documents"][i] or ""
        kind = metadata.get("kind", "").upper()

        score = 1
        for kw in keywords:
            if kw in node_id.lower():
                score += kind_weights.get(kind, 1.0)

        if score == 0:
            score = 0.1

        combined_score_norm = round((float(metadata.get("combined", 0.0) / COMBINED_MAX)), 2)
        hybrid_score = score + combined_score_norm
        candidate_nodes.append((node_id, metadata, doc, hybrid_score))

    return candidate_nodes


async def _score_node(question: str, node, code_snippet_limit: int) -> int:
    """
    Scores usefulness of node based of snippet of it's code.

    Args:
        question: User question
        code_snippet_limit: Maximum characters per code snippet

    Returns:
        Score for node in range 1-5.
    """

    logger.info(f"Scoring node: {node[0]}")
    snippet = "\n".join(node[2].splitlines())[:code_snippet_limit]
    prompt = f"""
    You are a technical code analyzer. Your task is to rate 
    the relevance of a Scala or Java code based on given question.

    Question: '{question}'

    Scoring Criteria:
    1: The code has nothing to do with the question.
    2: Somewhat connected to main topic, but not relevant
    3: Moderately appropriate for question, but full code should be relevant
    4: Not perfect, but relevant to question.
    5: Perfectly answers the question.

    Return only score in range 1-5. Do not include any text, reasoning, labels, or formatting.

    Example Correct Response:
    3

    Code fragment to analyze:
    {snippet}
    """
    try:
        answer = await call_llm(prompt)
        logger.debug(f"Answer: {answer}")
        match = re.search(r"([1-5])", str(answer))
        if match:
            return int(match.group(1))

        logger.warning(f"Could not get score for node: {node[0]}")
        return 1
    except Exception as e:
        logger.exception(f"ERROR in scoring node: {e}")
        return 1


def expand_node_with_neighbors(
    node_id: str,
    metadata: Dict[str, Any],
    collection: Any,
    seen_nodes: Set[str],
    max_neighbors: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Expands a node with its related neighbors.

    Args:
        node_id: Node identifier
        metadata: Node metadata
        collection: Chroma collection
        seen_nodes: Set of already seen node IDs
        max_neighbors: Maximum number of neighbors to fetch

    Returns:
        List of neighbor nodes with scores
    """
    related_entities_raw = metadata.get("related_entities")

    related_entities = json.loads(related_entities_raw)
    neighbors_to_fetch = []

    for entity_id, _ in related_entities:
        if entity_id not in seen_nodes:
            seen_nodes.add(entity_id)
            neighbors_to_fetch.append(entity_id)
            if len(neighbors_to_fetch) >= max_neighbors:
                break

    if not neighbors_to_fetch:
        return []

    neighbors = collection.get(ids=neighbors_to_fetch, include=["metadatas", "documents"])

    combined_neighbors = list(zip(neighbors["ids"], neighbors["metadatas"], neighbors["documents"]))
    combined_neighbors = sorted(
        combined_neighbors, key=lambda x: x[1].get("combined", 0), reverse=True
    )

    neighbors_data = []
    for neighbor_id, neighbor_metadata, neighbor_doc in combined_neighbors:
        if neighbor_id in seen_nodes:
            continue
        if metadata.get("kind") == "CLASS" and is_child_of(node_id, neighbor_id):
            continue
        neighbors_data.append(
            (-1, {"node": neighbor_id, "metadata": neighbor_metadata, "code": neighbor_doc})
        )

    return neighbors_data


async def get_general_nodes_context(
    question: str,
    analysis: IntentAnalysis,
    model_name: str,
    collection: Any,
    code_snippet_limit: int = 1200,
    **params,
) -> None | tuple[list[Any], str] | list[tuple[int, dict[str, Any]]]:
    """
    Retrieves top nodes for a general question using LLM-guided filtering.

    Args:
        question: Natural-language user question
        analysis: Analysis of user question
        model_name: Name of LLM model used for finding context
        collection: Chroma collection handle
        code_snippet_limit: Max characters per snippet for LLM
        **params:
            top_k: Number of nodes to get context from
            max_neighbors: How many neighbors of each top node to get

    Returns:
        Top nodes with metadata and code
    """
    top_k = params.get("top_k", 5)
    max_neighbors = params.get("max_neighbors", 3)
    logger.info(f"TOP NODES: {top_k}, Max neighbors: {max_neighbors}")

    start_time = time.time()
    logger.debug("Using LLM-based general_question filtering")
    kind_weights = {
        "CLASS": 2.0,
        "INTERFACE": 1.8,
        "TRAIT": 1.6,
        "ENUM": 1.5,
        "CONSTRUCTOR": 1.2,
        "METHOD": 1.0,
        "TYPE": 0.7,
        "TYPE_PARAMETER": 0.6,
        "OBJECT": 0.5,
    }

    kinds = [k.upper() for k in params.get("kinds", [])]
    keywords = [kw.lower() for kw in params.get("keywords", [])]
    logger.debug(f"kinds: {kinds}, keywords: {keywords}")

    if len(kinds) == 1:
        where_clause = {"kind": kinds[0]}
    elif len(kinds) > 1:
        where_clause = {"kind": {"$in": kinds}}
    else:
        where_clause = None

    all_nodes = collection.get(include=["metadatas", "documents"], where=where_clause)
    candidate_nodes = _filter_candidates(all_nodes, keywords, kind_weights)

    if not candidate_nodes:
        logger.debug("No candidates found, selecting fallback top-5 by combined score")
        fallback_nodes = sorted(
            zip(all_nodes["ids"], all_nodes["metadatas"], all_nodes["documents"]),
            key=lambda x: float(x[1].get("combined", 0.0)),
            reverse=True,
        )[:top_k]
        return [
            (1, {"node": nid, "metadata": meta, "code": doc}) for nid, meta, doc in fallback_nodes
        ]

    candidates_sorted = sorted(candidate_nodes, key=lambda x: x[3], reverse=True)[: top_k * 2]
    top_nodes = []
    seen_nodes = set()
    logger.info(f"Found {len(candidates_sorted)} candidates")
    logger.debug(f"Candidate: {[n[1]['node'] for n in candidates_sorted]}")
    for candidate in candidates_sorted:
        node_id, metadata, doc, hybrid_score = candidate

        try:
            score = await _score_node(question, candidate, code_snippet_limit)
        except Exception as e:
            logger.exception(f"Error while scoring nodes: {e}")

        if score >= 3 and node_id not in seen_nodes:
            top_nodes.append(
                (
                    hybrid_score + score * 2,
                    {"node": node_id, "metadata": metadata, "code": doc},
                )
            )
            seen_nodes.add(node_id)

    top_nodes = sorted(top_nodes, key=lambda x: x[0], reverse=True)[:top_k]
    logger.info(f"Top nodes length: {len(top_nodes)}")
    top_nodes_length = len(top_nodes)

    final_top_nodes = top_nodes.copy()
    seen_nodes = set(n[1]["node"] for n in top_nodes)
    for score, node_data in top_nodes:
        neighbor_nodes = expand_node_with_neighbors(
            node_data["node"], node_data["metadata"], collection, seen_nodes, max_neighbors
        )
        for score_offset, neighbor_data in neighbor_nodes:
            if neighbor_data["node"] not in seen_nodes:
                final_top_nodes.append((score_offset, neighbor_data))
                seen_nodes.add(neighbor_data["node"])

    for _, node_data in top_nodes:
        if node_data["metadata"].get("kind") == "CLASS":
            class_name = node_data["metadata"].get("label")
            usage_nodes = find_usage_nodes(
                collection, class_name, max_results=max_usage_nodes_for_context
            )
            for u_score, u_node_id, u_doc, u_metadata in usage_nodes:
                if u_node_id in seen_nodes:
                    continue
                final_top_nodes.append(
                    (u_score - 1, {"node": u_node_id, "metadata": u_metadata, "code": u_doc})
                )
                seen_nodes.add(u_node_id)

    logger.debug(f"Found {len(final_top_nodes)} top_nodes")
    logger.debug(f"TOP NODES from general_question: {[n[1]['node'] for n in final_top_nodes]}")

    category = getattr(analysis, "primary_intent", IntentCategory.GENERAL)
    if hasattr(category, "value"):
        category = category.value
    confidence = getattr(analysis, "confidence", 0.5)

    full_context = build_context(
        final_top_nodes,
        category,
        confidence,
        top_nodes_length,
        question=question,
        target=None,
    )

    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000
    logger.debug(f"Completed in: {elapsed_ms:.1f}ms")
    return final_top_nodes, full_context or "<NO CONTEXT FOUND>"
