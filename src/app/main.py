import asyncio
import logging
import time

import httpx

from clients.chroma_client import get_collection
from src.core.config import (
    CODEBERT_MODEL_NAME,
    HTTP_TIMEOUT,
    RAG_TIMEOUT,
)
from src.core.intent_analyzer import get_intent_analyzer
from src.core.prompt import build_prompt
from testing.token_counter import count_tokens

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


timeout: httpx.Timeout = httpx.Timeout(HTTP_TIMEOUT)
client: httpx.AsyncClient = httpx.AsyncClient(timeout=timeout)


def warm_up_models() -> None:
    """
    Preloads key models to reduce startup latency.
    Logs progress, total warm-up time, and any errors encountered.
    Does not raise exceptions.
    """
    logger.info("Warming up models...")
    start_time = time.time()
    try:
        from src.core.rag_optimization import _get_cached_model
        from src.graph.model_cache import get_graph_model

        get_graph_model()
        logger.info("Graph model loaded.")
        _get_cached_model()
        logger.info("CodeBERT model loaded.")
        warmup_time = time.time() - start_time
        logger.info(f"RAG models loaded in {warmup_time:.2f}s")
    except Exception as e:
        logger.error(f"Model warmup failed: {e}")


async def retrieve_context(question: str, node_func, params: dict):
    """
    Core function for RAG-style context retrieval.

    Performs intent analysis, retrieves relevant nodes using the provided
    function, builds a prompt, and counts tokens.

    Args:
        question: User's question
        node_func: Function to retrieve nodes (specific, top, general, or related)
        params: Additional parameters for the node function

    Returns:
        dict with fields:
            - status: "success" or "error"
            - context: Retrieved context or error message
            - prompt: Built prompt for LLM
            - matches: Number of matching nodes found
            - total_time: Processing time in seconds
            - tokens: Token counts for context and prompt
    """
    start_time = time.time()
    logger.info(f"Received question: {question}")

    if params is None:
        params = {}

    try:
        analyzer = get_intent_analyzer()
        intent_start_time = time.time()
        user_intent_analysis = analyzer.enhanced_classify_question(question)
        user_intent = {"primary_intent": user_intent_analysis.primary_intent.value}

        intent_time = time.time() - intent_start_time
        logger.info(f"User intent analysis: {intent_time:.3f}s - {user_intent}")

        rag_start_time = time.time()
        matches, context = await asyncio.wait_for(
            node_func(
                question,
                user_intent_analysis,
                model_name=CODEBERT_MODEL_NAME,
                collection=get_collection("scg_embeddings"),
                **params,
            ),
            timeout=RAG_TIMEOUT,
        )
        rag_time = time.time() - rag_start_time
        logger.info(f"RAG processing: {rag_time:.3f}s, found {len(matches)} matches")

        prompt_start_time = time.time()
        prompt = build_prompt(
            question=question,
            context=context,
            intent=user_intent,
        )
        prompt_time = time.time() - prompt_start_time
        logger.info(f"Intent-aware prompt built: {prompt_time:.3f}s, length: {len(prompt)} chars")

        if logger.level <= logging.DEBUG:
            logger.debug("Generated prompt:")
            logger.debug(prompt)

        if not isinstance(context, str):
            context = str(context)

        logger.info(f"Context: {context}")
        context_tokens = count_tokens(context, "gemini")
        prompt_tokens = count_tokens(prompt, "gemini")
        logger.info(f"Tokens: context: {context_tokens}, prompt: {prompt_tokens}")

        total_time = time.time() - start_time
        logger.info(f"Total time: {total_time:.3f}s")

        return {
            "status": "success",
            "context": context,
            "prompt": prompt,
            "matches": len(matches) if matches else 0,
            "total_time": total_time,
            "tokens": {
                "context": context_tokens,
                "prompt": prompt_tokens,
            },
        }

    except Exception as e:
        return {"status": "error", "context": f"Error: {str(e)}", "matches": 0}

