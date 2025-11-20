import json
import re
from typing import Any, Dict, List, Set, Tuple

from loguru import logger

from src.clients.llm_client import call_llm
from src.graph.usage_finder import find_usage_nodes

max_usage_nodes_for_context = 5


def _filter_candidates(
    all_nodes: Dict[str, Any], kinds: Set[str], keywords: List[str], kind_weights: Dict[str, float]
) -> List[Tuple[str, Dict[str, Any], str, float]]:
    """
    Filters and scores candidate nodes based on LLM analysis.

    Args:
        all_nodes: All nodes from collection
        kinds: Set of relevant node kinds (CLASS, METHOD, etc.)
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

        if kinds and kind not in kinds:
            continue

        score = 1 if (not kinds or kind in kinds) else 0
        for kw in keywords:
            if kw in node_id.lower():
                score += kind_weights.get(kind, 1.0)

        if score == 0:
            score = 0.1

        combined_score = float(metadata.get("combined", 0.0))
        hybrid_score = score * 1000 + combined_score
        candidate_nodes.append((node_id, metadata, doc, hybrid_score))

    return candidate_nodes


async def _score_batch(
    question: str, batch: List[Tuple[str, Dict[str, Any], str, float]], code_snippet_limit: int
) -> List[Dict[str, Any]]:
    """
    Scores a batch of candidates using LLM.

    Args:
        question: User question
        batch: Batch of candidate nodes
        code_snippet_limit: Maximum characters per code snippet

    Returns:
        List of scored items from LLM
    """
    code_snippets_map = []
    for node_id, meta, doc, hybrid_score in batch:
        snippet = "\n".join(doc.splitlines())[:code_snippet_limit]
        code_snippets_map.append({"id": node_id, "code": snippet})

    prompt = f"""
Question: '{question}'

Rate each code fragment from 1 to 5:
1 = not relevant at all
3 = moderately relevant, the full code should help answer the question
5 = directly answers the question

Return JSON: [{{"id": "node_id", "score": 3}}, ...]

No comments or explanations.

Code fragments:
{json.dumps(code_snippets_map, indent=2)}
"""

    answer = await call_llm(prompt)
    logger.debug("LLM batch scoring response")

    clean_answer = re.sub(r"```(?:json)?", "", answer).strip()
    try:
        return json.loads(clean_answer)
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


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
    neighbors_data = []

    related_entities_str = metadata.get("related_entities", "")
    try:
        related_entities = (
            json.loads(related_entities_str)
            if isinstance(related_entities_str, str)
            else related_entities_str
        )
    except (json.JSONDecodeError, TypeError):
        related_entities = []

    neighbors_to_fetch = [nid for nid in related_entities[:max_neighbors] if nid not in seen_nodes]

    if not neighbors_to_fetch:
        return neighbors_data

    neighbors = collection.get(ids=neighbors_to_fetch, include=["metadatas", "documents"])
    for j in range(len(neighbors["ids"])):
        neighbor_id = neighbors["ids"][j]
        if neighbor_id in seen_nodes:
            continue
        neighbor_metadata = neighbors["metadatas"][j]
        neighbor_kind = neighbor_metadata.get("kind", "")
        neighbor_doc = neighbors["documents"][j] or ""
        if (
            metadata.get("kind") == "CLASS"
            and (neighbor_kind == "METHOD" or neighbor_kind == "VARIABLE")
            and str(neighbor_id).startswith(f"{node_id}.")
        ):
            continue
        neighbors_data.append(
            (-1, {"node": neighbor_id, "metadata": neighbor_metadata, "code": neighbor_doc})
        )
    return neighbors_data


async def general_question(
    question: str,
    collection: Any,
    top_k: int = 5,
    max_neighbors: int = 3,
    code_snippet_limit: int = 800,
    batch_size: int = 5,
) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Retrieves top nodes for a general question using LLM-guided filtering.

    Args:
        question: Natural-language user question
        collection: Chroma collection handle
        top_k: Number of final nodes to return
        max_neighbors: Max related neighbors per node
        code_snippet_limit: Max characters per snippet for LLM
        batch_size: Number of candidates scored per batch

    Returns:
        Top nodes with metadata and code
    """
    kind_weights = {
        "CLASS": 2.0,
        "INTERFACE": 1.8,
        "METHOD": 1.0,
        "CONSTRUCTOR": 1.2,
        "VARIABLE": 0.8,
        "PARAMETER": 0.5,
    }
    classification_prompt = f"""
    User question: "{question}"
    
    Your task:
    1. Determine which node types (CLASS, METHOD, VARIABLE, PARAMETER, CONSTRUCTOR) 
        are most relevant
    2. Provide 5-10 keywords that should appear in node names
    
    Return ONLY valid JSON format:
    {{"kinds": ["CLASS", "METHOD"], "keywords": ["frontend","controller","view"]}}
    
    No comments, only JSON.
    """
    analysis = await call_llm(classification_prompt)
    logger.debug(f"LLM analysis: {analysis}")
    try:
        analysis = json.loads(analysis)
    except (json.JSONDecodeError, TypeError):
        analysis = {"kinds": [], "keywords": []}
    kinds = set([k.upper() for k in analysis.get("kinds", [])])
    keywords = [kw.lower() for kw in analysis.get("keywords", [])]
    all_nodes = collection.get(include=["metadatas", "documents"])
    logger.debug(f"kinds: {kinds}, keywords: {keywords}")
    candidate_nodes = _filter_candidates(all_nodes, kinds, keywords, kind_weights)
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
    for i in range(0, len(candidates_sorted), batch_size):
        batch = candidates_sorted[i : i + batch_size]
        scores = await _score_batch(question, batch, code_snippet_limit)
        logger.debug(f"LLM scores: {scores}")
        for s in scores:
            node_id = s.get("id")
            score = int(s.get("score", 0))
            if score >= 3:
                node_tuple = next((c for c in batch if c[0] == node_id), None)
                if node_tuple:
                    _, metadata, doc, _ = node_tuple
                    if node_id not in seen_nodes:
                        top_nodes.append(
                            (score, {"node": node_id, "metadata": metadata, "code": doc})
                        )
                        seen_nodes.add(node_id)
                    neighbor_nodes = expand_node_with_neighbors(
                        node_id, metadata, collection, seen_nodes, max_neighbors
                    )
                    for neighbor_score_offset, neighbor_data in neighbor_nodes:
                        top_nodes.append((score + neighbor_score_offset, neighbor_data))
                        seen_nodes.add(neighbor_data["node"])
    final_top_nodes = top_nodes.copy()
    for score, node_data in top_nodes:
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
    top_nodes = sorted(final_top_nodes, key=lambda x: x[0], reverse=True)[:top_k]
    logger.debug(f"TOP NODES from general_question: {[n[1]['node'] for n in top_nodes]}")
    return top_nodes
