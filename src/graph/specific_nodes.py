import time
from typing import Any, Dict, List, Tuple, Optional
from loguru import logger

from core.intent_analyzer import get_intent_analyzer
from core.models import IntentAnalysis
from graph.reranking import rerank_results
from graph.retrieval_utils import deduplicate_results, identify_target_entity, expand_usage_results, \
    expand_definition_neighbors
from graph.retriver import extract_key_value_pairs_simple
from graph.similar_node_optimization import get_graph_model


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
    top_k = params.get('top_k', 10)
    max_neighbors = params.get('max_neighbors', 2)
    logger.info(f"TOP K: {top_k}, Max neighbors: {max_neighbors}")
    try:
        from src.graph.generate_embeddings_graph import generate_embeddings_graph
        from src.graph.retriver import extract_key_value_pairs_simple
        from graph.retrieval_utils import query_embeddings, get_embedding_inputs

        try:
            from context.context_builder import build_context
        except ImportError:

            def build_context(nodes, category, confidence, question="", target_method=None):
                return "\n\n".join([node[1]["code"] for node in nodes[:5] if node[1]["code"]])

        analyzer = get_intent_analyzer()
        analysis_result = analyzer.enhanced_classify_question(question)
        analysis = {
            "category": analysis_result.primary_intent.value,
            "confidence": analysis_result.confidence,
            "scores": analysis_result.scores,
            "enhanced": analysis_result.enhanced,
        }
        logger.debug(f"Enhanced classification: {analysis}")

        pairs = await extract_key_value_pairs_simple(question)
        pairs = [(t.lower(), n.lower()) for t, n in pairs]

        logger.debug(f"Question: '{question}'")
        logger.debug(f"Extracted pairs: {pairs}")

        embeddings_input = get_embedding_inputs(pairs, question)
        logger.debug(f"Embedding input: {embeddings_input}")

        get_graph_model()

        query_embeddings_var = generate_embeddings_graph(embeddings_input, model_name)
        all_results = query_embeddings(collection, query_embeddings_var, embeddings_input)
        reranked_results = rerank_results(question, all_results, analysis)
        logger.debug(f"Reranked {len(reranked_results)} results")
        unique_results = deduplicate_results(reranked_results, len(embeddings_input) * top_k)
        target_entity = None
        if analyzer.is_usage_question(question):
            logger.debug("Usage question. Searching in related_entities")
            target_entity = identify_target_entity(unique_results)
            top_nodes = expand_usage_results(unique_results, collection, target_entity)
        else:
            top_nodes = expand_definition_neighbors(unique_results, collection, max_neighbors)
        logger.debug(f"Selected {len(top_nodes)} best nodes")

        category = analysis.get("category", "general")
        confidence = analysis.get("confidence", 0.5)

        logger.debug(
            f"Building context with category={category}, confidence={confidence},"
            f" target={target_entity}"
        )

        full_context = build_context(
            top_nodes, category, confidence, question=question, target_method=target_entity
        )

        logger.debug(f"Context built: {len(full_context)} chars")

        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        logger.debug(f"Completed in: {elapsed_ms:.1f}ms")

        return top_nodes, full_context or "<NO CONTEXT FOUND>"

    except Exception as e:
        logger.warning(f"Fallback do oryginalnej funkcji: {e}")
        from src.graph.retriver import similar_node
        return similar_node(question, model_name, top_k)


_general_fallback_cache: Optional[str] = None
_cache_timestamp: float = 0
