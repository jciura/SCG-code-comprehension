import asyncio
import logging
import time
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from clients.chroma_client import get_collection
from graph.general_query import get_general_nodes_context
from graph.related_entities import get_related_entities
from graph.specific_nodes import get_specific_nodes_context
from graph.top_nodes import get_top_nodes_context
from src.core.config import (
    CODEBERT_MODEL_NAME,
    HTTP_TIMEOUT,
    CORS_ORIGINS,
    RAG_TIMEOUT,
)
from src.core.intent_analyzer import get_intent_analyzer
from src.core.models import AskRequest


from src.core.prompt import build_prompt
from testing.token_counter import count_tokens



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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



@app.post("/ask_specific_nodes")
async def ask_specific_nodes(req: AskRequest):
    """
        Retrieve context for specific named entities.

        Args:
            req: Request with question and optional params

        Returns:
            Context from matched specific nodes
        """
    return await retrieve_context(req.question, get_specific_nodes_context, req.params)


@app.post("/ask_top_nodes")
async def ask_top_nodes(req: AskRequest):
    """
        Retrieve context from most important nodes in the graph.

        Args:
            req: Request with question and optional params

        Returns:
            Context from top-ranked nodes by importance
        """
    return await retrieve_context(req.question, get_top_nodes_context, req.params)


@app.post("/ask_general_question")
async def ask_general_question(req: AskRequest):
    """
        Retrieve general graph context.

        Args:
            req: Request with question and optional params

        Returns:
            Context from semantically similar nodes
        """
    return await retrieve_context(req.question, get_general_nodes_context, req.params)


@app.post("/list_related_entities")
async def list_related_entities(req: AskRequest):
    """
        Retrieve related entities from the knowledge graph.

        Args:
            req: Request with question and optional params

        Returns:
            Context with related entities and their relationships
        """
    return await retrieve_context(req.question, get_related_entities, req.params)


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
            }
        }

    except Exception as e:
        return {"status": "error", "context": f"Error: {str(e)}", "matches": 0}


@app.on_event("startup")
async def startup_event():
    """
    Application startup event handler.

    Initializes models, evaluator.
    Keeps the selected model in memory and performs a warm-up request to
    reduce first-call latency.

    Logs the status of each initialization step and reports any failures.
    """
    logger.info("Starting RAG application...")
    warm_up_models()
    logger.info("RAG application startup completed")


@app.on_event("shutdown")
async def shutdown():
    """
    Application shutdown event handler.

    Closes the shared HTTP client and logs shutdown completion.
    """
    logger.info("Shutting down RAG application...")
    await client.aclose()
    logger.info("HTTP client closed")
