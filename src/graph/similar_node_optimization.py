import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from src.clients.chroma_client import get_collection
from src.core.intent_analyzer import get_intent_analyzer
from src.graph.general_query import general_question
from src.graph.reranking import rerank_results
from src.graph.retrieval_utils import (
    deduplicate_results,
    expand_definition_neighbors,
    expand_usage_results,
    get_embedding_inputs,
    identify_target_entity,
)
from src.graph.top_nodes import find_top_nodes

_graph_model = None


def get_graph_model() -> bool:
    """
    Ensures the graph embedding model is initialized.

    Lazily loads the graph model utilities if not already loaded and logs progress.

    Returns:
        bool: Always True after ensuring initialization.
    """
    global _graph_model
    if _graph_model is None:
        logger.info("Loading graph model")
        logger.info("Graph model is ready")
    return True


@lru_cache(maxsize=1000)
def cached_question_question(question_hash: str) -> str:
    """
    Cached classification of a question.

    Uses a lightweight classifier to map a (hashed) question to a category.
    Results are memoized via `functools.lru_cache`.

    Args:
        question_hash (str): Hash or key representing the question to classify.

    Returns:
        str: Classified category label for the question.
    """
    from src.core.intent_analyzer import classify_question

    return classify_question(question_hash)


async def similar_node_fast(
    question: str, model_name: str = "microsoft/codebert-base", top_k: int = 20
) -> Tuple[List[Tuple[float, Dict[str, Any]]], str]:
    """
    Fast path for retrieving similar nodes and building context.

    This is the main orchestrator that:
    1. Analyzes intent
    2. Routes to appropriate handler (top/general/exception/standard)
    3. Reranks results
    4. Expands with neighbors/usage
    5. Builds final context

    Args:
        question: User's natural-language question
        model_name: Embedding model identifier
        top_k: Base number of results per query embedding

    Returns:
        Tuple of (retrieved nodes, context string)
    """
    start_time = time.time()

    try:
        from src.graph.generate_embeddings_graph import generate_embeddings_graph
        from src.graph.retriver import extract_key_value_pairs_simple

        try:
            from context.context_builder import build_context
        except ImportError:

            def build_context(nodes, category, confidence, question="", target_method=None):
                return "\n\n".join([node[1]["code"] for node in nodes[:5] if node[1]["code"]])

        collection = get_collection("scg_embeddings")

        pairs = await extract_key_value_pairs_simple(question)
        logger.debug(f"Question: '{question}'")
        logger.debug(f"Extracted pairs: {pairs}")

        embeddings_input = get_embedding_inputs(pairs, question)
        logger.debug(f"Embedding input: {embeddings_input}")

        get_graph_model()

        analyzer = get_intent_analyzer()
        analysis_result = analyzer.enhanced_classify_question(question)
        analysis = {
            "category": analysis_result.primary_intent.value,
            "confidence": analysis_result.confidence,
            "scores": analysis_result.scores,
            "enhanced": analysis_result.enhanced,
        }
        logger.debug(f"Enhanced classification: {analysis}")

        if analysis["category"] == "top" and analysis["confidence"] > 0.6:
            logger.debug("Finding top classes or methods")
            top_nodes = await find_top_nodes(question, collection)
            context = " ".join(
                f"{node.get('metadata', {}).get('label', '')} - {node.get('metric_value'):.2f}"
                for node in top_nodes
            )
            logger.debug(f"Top query context: {context}")

            end_time = time.time()
            elapsed_ms = (end_time - start_time) * 1000
            logger.debug(f"Completed in: {elapsed_ms:.1f}ms")
            return top_nodes, context or "<NO CONTEXT FOUND>"

        if not pairs or (analysis["category"] == "general" and analysis["confidence"] > 0.6):
            if analysis["category"] == "exception":
                logger.debug("EXCEPTION category detected - forcing embeddings search")
                embeddings_input = [question]
                query_embeddings = generate_embeddings_graph(embeddings_input, model_name)
                query_result = collection.query(
                    query_embeddings=[query_embeddings[0].tolist()],
                    n_results=top_k * 2,
                    include=["embeddings", "metadatas", "documents", "distances"],
                )

                all_results = []
                for i in range(len(query_result["ids"][0])):
                    score = 1 - query_result["distances"][0][i]
                    node_id = query_result["ids"][0][i]
                    metadata = query_result["metadatas"][0][i]
                    code = query_result["documents"][0][i]
                    all_results.append(
                        (score, {"node": node_id, "metadata": metadata, "code": code})
                    )

                reranked_results = rerank_results(question, all_results, analysis)
                seen = set()
                unique_results = []
                for score, node in reranked_results:
                    if node["node"] not in seen:
                        unique_results.append((score, node))
                        seen.add(node["node"])

                top_nodes = unique_results[:10]
                logger.debug(f"EXCEPTION: found {len(top_nodes)} nodes via embeddings")

            else:
                logger.debug("Using LLM-based general_question filtering")
                top_nodes = await general_question(question, collection, top_k=5, max_neighbors=2)

            category = analysis.get("category", "general")
            confidence = analysis.get("confidence", 0.5)
            full_context = build_context(
                top_nodes, category, confidence, question=question, target_method=None
            )

            end_time = time.time()
            elapsed_ms = (end_time - start_time) * 1000
            logger.debug(f"Completed in: {elapsed_ms:.1f}ms")
            return top_nodes, full_context or "<NO CONTEXT FOUND>"

        query_embeddings = generate_embeddings_graph(embeddings_input, model_name)
        all_results = query_embeddings(collection, query_embeddings, embeddings_input)
        reranked_results = rerank_results(question, all_results, analysis)
        logger.debug(f"Reranked {len(reranked_results)} results")
        unique_results = deduplicate_results(reranked_results, len(embeddings_input) * top_k)
        target_entity = None
        if analyzer.is_usage_question(question):
            logger.debug("Usage question. Searching in related_entities")
            target_entity = identify_target_entity(unique_results)
            top_nodes = expand_usage_results(unique_results, collection, target_entity)
        else:
            top_nodes = expand_definition_neighbors(unique_results, collection)
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
        logger.warning(f"Fallback to original function: {e}")
        from src.graph.retriver import similar_node

        return similar_node(question, model_name, top_k)


_general_fallback_cache: Optional[str] = None
_cache_timestamp: float = 0
