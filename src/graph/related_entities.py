import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from loguru import logger

from core.models import IntentAnalysis
from graph.NeighborTypeEnum import NeighborTypeEnum
from graph.RelationTypes import RelationTypes
from graph.generate_embeddings_graph import generate_embeddings_graph
from graph.model_cache import get_graph_model
from graph.retrieval_utils import get_embedding_inputs, query_embeddings


async def get_related_entities(
        question: str,
        analysis: IntentAnalysis,
        model_name: str,
        collection: Any,
        **params,
) -> Tuple[List[Tuple[float, Dict[str, Any]]], str]:
    """
    Find nodes included in user question

    Args:
        question: Natural-language question
        analysis: Analysis of user's question
        model_name: Name of LLM model used for finding context
        collection: Chroma collection handle
        **params:
            top_k: Maximum number of top nodes to get context from
            max_neighbors: Maximum number of neighbors for each top node

    Returns:
        Top nodes with metadata and metric values
    """
    limit = params.get("limit", 5)
    neighbor_types = params.get("neighbor_types", ["ANY"])
    if isinstance(neighbor_types, str):
        neighbor_types = [neighbor_types]
    neighbor_types = [NeighborTypeEnum[neighbor_type.upper()] for neighbor_type in neighbor_types]
    logger.info(f"Limit: {limit}, Neighbor types: {neighbor_types}")

    pairs = params.get("pairs", [])
    pairs = [(t.lower(), n.lower()) for t, n in pairs]

    relation_types = params.get("relation_types", ["ANY"])
    if isinstance(relation_types, str):
        neighbor_types = [neighbor_types]
    relation_types = [RelationTypes[relation_type.upper()] for relation_type in relation_types]


    logger.debug(f"Question: '{question}'")
    logger.debug(f"Extracted pairs: {pairs}")

    embeddings_input = get_embedding_inputs(pairs, question)
    logger.debug(f"Embedding input: {embeddings_input}")

    get_graph_model()

    query_embeddings_var = generate_embeddings_graph(embeddings_input, model_name)
    results = query_embeddings(collection, query_embeddings_var, embeddings_input, pairs)

    context = ""

    neighbor_type_filter = [
        nt.value.upper()
        for nt in neighbor_types
        if nt != NeighborTypeEnum.ANY
    ]

    for result in results:
        node_id = result[1]["node"]
        metadata = result[1]["metadata"]
        related_entities_raw = metadata.get("related_entities")

        if not related_entities_raw:
            continue

        related_entities = json.loads(related_entities_raw)
        relations_by_entity = defaultdict(list)

        for entity_id, relation in related_entities:
            if RelationTypes[relation.upper()] in relation_types or RelationTypes.ANY in relation_types:
                relations_by_entity[entity_id].append(relation)

        neighbors_to_fetch = list(relations_by_entity.keys())

        if len(neighbors_to_fetch) == 0:
            return [], "No neighbors to fetch"

        if NeighborTypeEnum.ANY in neighbor_types:
            neighbors = collection.get(
                ids=neighbors_to_fetch,
                include=["metadatas", "documents"],
            )
        else:
            neighbors = collection.get(
                ids=neighbors_to_fetch,
                include=["metadatas", "documents"],
                where={"kind": {"$in": neighbor_type_filter}},
            )

        combined_neighbors = list(zip(neighbors["ids"], neighbors["metadatas"], neighbors["documents"]))

        neighbors_sorted = sorted(
            combined_neighbors,
            key=lambda x: x[1].get("combined", 0),
            reverse=True
        )

        if limit == "all":
            neighbors_sorted = neighbors_sorted
        else:
            neighbors_sorted = neighbors_sorted[:limit]

        logger.info(f"Neighbors: {len(neighbors_sorted)}")
        context += f"Neighbors (sorted by combined metric importance) of Node: {node_id}, with name: {metadata['label']} kind: {metadata['kind']} located in: {metadata['uri']}:\n"

        for i, (node_id, neighbors_metadata, _) in enumerate(neighbors_sorted):
            relations = relations_by_entity.get(node_id)
            context += f"{i + 1}.Node_id: {node_id}, name: {neighbors_metadata.get('label', '')}, kind: {neighbors_metadata.get('kind', '')}, uri: {neighbors_metadata.get('uri', '')}, relations: {', '.join(relations)}\n"

    logger.info(f"Context: {context}")
    return [], context or "<NO CONTEXT FOUND>"
