import time
from typing import Any, Dict, List, Tuple

from loguru import logger

from core.models import IntentAnalysis, IntentCategory
from graph.model_cache import get_graph_model
from graph.NeighborTypeEnum import NeighborTypeEnum
from graph.RelationTypes import RelationTypes
from graph.reranking import rerank_results
from graph.retrieval_utils import (
    deduplicate_results,
    expand_definition_neighbors,
    expand_usage_results,
    identify_target_entity,
)


async def get_specific_nodes_context(
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
    start_time = time.time()
    top_k = params.get("top_k", 10)
    max_neighbors = params.get("max_neighbors", 2)
    neighbor_types = params.get("neighbor_types", "ANY")
    if isinstance(neighbor_types, str):
        neighbor_types = [neighbor_types]
    neighbor_types = [NeighborTypeEnum[type.upper()] for type in neighbor_types]
    pairs = params.get("pairs", [])
    relation_types = params.get("relation_types", [])
    if isinstance(relation_types, str):
        relation_types = [relation_types]
    relation_types = [RelationTypes[type.upper()] for type in relation_types]
    logger.info(
        f"TOP K: {top_k}, Max neighbors: {max_neighbors}, "
        f"Neighbor types: {neighbor_types}, pairs: {pairs}, "
        f"relation_types: {relation_types}"
    )
    try:
        from graph.retrieval_utils import get_embedding_inputs, query_embeddings
        from src.graph.generate_embeddings_graph import generate_embeddings_graph

        try:
            from context.context_builder import build_context
        except ImportError:

            def build_context(nodes, category, confidence, top_nodes_limit=1, question="", target=None):
                return "\n\n".join([node[1]["code"] for node in nodes[:5] if node[1]["code"]])

        logger.debug(f"Question: '{question}'")
        logger.debug(f"Extracted pairs: {pairs}")

        embeddings_input = get_embedding_inputs(pairs, question)
        logger.debug(f"Embedding input: {embeddings_input}")

        get_graph_model()

        query_embeddings_var = generate_embeddings_graph(embeddings_input, model_name)
        all_results = query_embeddings(collection, query_embeddings_var, embeddings_input, pairs)
        reranked_results = rerank_results(question, all_results, analysis)
        logger.debug(f"Reranked {len(reranked_results)} results")
        unique_results = deduplicate_results(reranked_results, len(embeddings_input) * top_k)
        target_entity = None
        category = getattr(analysis, "primary_intent", IntentCategory.GENERAL)
        confidence = getattr(analysis, "confidence", 0.5)

        if category == IntentCategory.USAGE:
            logger.debug("Usage question. Searching in related_entities")
            target_entity = identify_target_entity(unique_results)
            top_nodes = expand_usage_results(unique_results, collection, target_entity)
        else:
            top_nodes = expand_definition_neighbors(
                unique_results, collection, max_neighbors, neighbor_types, relation_types
            )
        logger.debug(f"Selected {len(top_nodes)} best nodes")

        logger.debug(
            f"Building context with category={category}, confidence={confidence},"
            f" target={target_entity}"
        )

        full_context = build_context(
            top_nodes,
            category,
            confidence,
            len(pairs),
            question=question,
            target=target_entity,
        )

        logger.debug(f"Context built: {len(full_context)} chars")

        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        logger.debug(f"Completed in: {elapsed_ms:.1f}ms")

        return top_nodes, full_context or "<NO CONTEXT FOUND>"

    except Exception as e:
        logger.warning(f"Error occured during retrieving specific nodes: {e}")
        return None
