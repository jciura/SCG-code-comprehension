import json
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

from graph.NeighborTypeEnum import NeighborTypeEnum
from graph.RelationTypes import RelationTypes
from src.graph.usage_finder import find_usage_nodes

debug_results_limit = 5
max_top_nodes_for_usage = 7
max_usage_nodes_for_context = 5


def get_embedding_inputs(pairs: List[Tuple[str, str]], question: str) -> List[str]:
    """
    Converts extracted pairs to embedding input strings.

    Args:
        pairs: List of (type, name) pairs extracted from question
        question: Original question string

    Returns:
        List of strings to generate embeddings for
    """
    embeddings_input = []
    for key, value in pairs:
        embeddings_input.append(f"{key.lower()} {value.lower()}" if key else value)
    if not embeddings_input:
        embeddings_input = [question]
    return embeddings_input


def query_embeddings(
    collection: Any,
    query_embeddings: List[Any],
    embeddings_input: List[str],
    pairs: List[Tuple[str, str]],
) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Queries ChromaDB with embeddings and returns results.

    Args:
        collection: Chroma collection handle
        query_embeddings: List of embedding vectors
        embeddings_input: Original input strings (for logging)

    Returns:
        List of (score, node_data) tuples
    """
    all_results = []
    batch_embeddings = [emb.tolist() for emb in query_embeddings]

    for i, emb in enumerate(batch_embeddings):
        if len(batch_embeddings) > 1:
            logger.debug(f"  Query {i + 1}/{len(batch_embeddings)}: '{embeddings_input[i]}'")
        else:
            logger.debug("Simple query with 1 embedding")

        query_result = collection.query(
            query_embeddings=[emb],
            n_results=1,
            include=["embeddings", "metadatas", "documents", "distances"],
            where={"kind": pairs[i][0].upper()},
        )
        for j in range(len(query_result["ids"][0])):
            score = 1 - query_result["distances"][0][j]
            node_id = query_result["ids"][0][j]
            metadata = query_result["metadatas"][0][j]
            code = query_result["documents"][0][j]
            if j < debug_results_limit:
                raw_distance = query_result["distances"][0][j]
                calculated_score = 1 - raw_distance
                logger.debug(f"Result {j + 1}:")
                logger.debug(f"Node ID: {node_id}")
                logger.debug(f"Raw distance: {raw_distance:.4f}")
                logger.debug(f"Calculated score: {calculated_score:.4f}")
                logger.debug(f"Label: {metadata.get('label', 'NO_LABEL')}")
                logger.debug(f"Kind: {metadata.get('kind', 'NO_KIND')}")
                logger.debug("")
            all_results.append((score, {"node": node_id, "metadata": metadata, "code": code}))

    logger.debug(f"Collected {len(all_results)} results from ChromaDB")
    return all_results


def deduplicate_results(
    results: List[Tuple[float, Dict[str, Any]]], max_results: int
) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Deduplicates results by node ID.

    Args:
        results: List of (score, node_data) tuples
        max_results: Maximum number of unique results to return

    Returns:
        Deduplicated list of results
    """
    logger.debug(f"Deduplicating: removing duplicates from {len(results)} results")
    seen: Set[str] = set()
    unique_results = []
    for score, node in results:
        node_id = node["node"]
        if node_id not in seen:
            unique_results.append((score, node))
            seen.add(node_id)
            if len(unique_results) >= max_results:
                break
    logger.debug(f"After deduplication: {len(unique_results)} unique results")
    return unique_results


def identify_target_entity(unique_results: List[Tuple[float, Dict[str, Any]]]) -> Optional[str]:
    """
    Identifies target entity from best match for usage queries.

    Args:
        unique_results: List of unique results

    Returns:
        Target entity name or None
    """
    if not unique_results:
        return None

    best_match = unique_results[0]
    node_id = best_match[1]["node"]
    metadata = best_match[1]["metadata"]
    logger.debug(f"Using ORIGINAL best match from ChromaDB: {node_id}")

    if metadata.get("kind") == "METHOD":
        target_entity = metadata.get("label")
        logger.debug(f"Target method identified: {target_entity}")
        return target_entity
    elif metadata.get("kind") == "CLASS":
        target_entity = metadata.get("label")
        logger.debug(f"Target class identified: {target_entity}")
        return target_entity
    elif "." in node_id:
        parts = node_id.split(".")
        for part in reversed(parts):
            if part and part[0].isupper():
                logger.debug(f"Target class identified from node_id: {part}")
                return part
    return None


def is_child_of(node_id_parent, node_id_candidate):
    """
    Check whether a node identifier is a child of another node.

    A child node is identified by sharing the same base identifier
    before '#' or '?' separators.
    """
    for sep in ["#", "?"]:
        if sep in node_id_candidate:
            parent, _ = node_id_candidate.split(sep, 1)
            if parent == node_id_parent:
                return True
    return node_id_candidate == node_id_parent


def expand_usage_results(
    unique_results: List[Tuple[float, Dict[str, Any]]],
    collection: Any,
    target_entity: Optional[str],
) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Expands results with usage nodes.

    Args:
        unique_results: List of unique results
        collection: Chroma collection handle
        target_entity: Target entity name

    Returns:
        Expanded list with usage nodes
    """
    top_nodes = unique_results[:1]
    if target_entity:
        usage_nodes = find_usage_nodes(
            collection, target_entity, max_results=max_usage_nodes_for_context
        )
        for score, node_id, doc, metadata in usage_nodes:
            top_nodes.append((score, {"node": node_id, "metadata": metadata, "code": doc}))
        logger.debug(f"Added {len(usage_nodes)} usage nodes to results")
    return top_nodes[:max_top_nodes_for_usage]


def expand_definition_neighbors(
    unique_results: List[Tuple[float, Dict[str, Any]]],
    collection: Any,
    max_neighbors: int,
    neighbor_types: List[NeighborTypeEnum] = [NeighborTypeEnum.ANY],
    relation_types: List[RelationTypes] = [RelationTypes.ANY],
) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Expands definition results with neighbors if single object query.

    Args:
        unique_results: List of unique results
        collection: Chroma collection handle
        max_neighbors: Number of neighbors of each top node to get


    Returns:
        Expanded list with neighbor nodes
    """
    top_nodes = unique_results[: len(unique_results)]
    current_nodes = top_nodes.copy()
    added_nodes = {node_data["node"] for _, node_data in top_nodes}

    for node_score, node_data in current_nodes:
        node_id = node_data["node"]
        metadata = node_data["metadata"]
        parent_kind = metadata.get("kind", "")

        related_entities_raw = metadata.get("related_entities")

        if not related_entities_raw:
            continue

        related_entities = json.loads(related_entities_raw)

        neighbors_to_fetch = []
        seen = set()

        for entity_id, relation in related_entities:
            if entity_id not in seen and (
                RelationTypes[relation.upper()] in relation_types
                or RelationTypes.ANY in relation_types
            ):
                seen.add(entity_id)
                neighbors_to_fetch.append(entity_id)

        neighbors_added = 0

        if not neighbors_to_fetch:
            continue

        neighbors = collection.get(ids=neighbors_to_fetch, include=["metadatas", "documents"])

        combined_neighbors = list(
            zip(neighbors["ids"], neighbors["metadatas"], neighbors["documents"])
        )
        combined_neighbors = sorted(
            combined_neighbors, key=lambda x: x[1].get("combined", 0), reverse=True
        )

        for neighbor_id, neighbor_metadata, neighbor_doc in combined_neighbors:
            if neighbors_added >= max_neighbors:
                break

            neighbor_kind = neighbor_metadata.get("kind", "")

            if (
                NeighborTypeEnum.ANY not in neighbor_types
                and NeighborTypeEnum[neighbor_kind.upper()] not in neighbor_types
            ):
                continue

            if (
                parent_kind == "CLASS" and is_child_of(node_id, neighbor_id)
            ) or neighbor_id in added_nodes:
                continue

            top_nodes.append(
                (
                    node_score,
                    {
                        "node": neighbor_id,
                        "metadata": neighbor_metadata,
                        "code": neighbor_doc or "",
                    },
                )
            )
            added_nodes.add(neighbor_id)
            neighbors_added += 1

        logger.info(f"Added nodes: {added_nodes}")

    return top_nodes
