import asyncio
import json
import logging
import os
import time

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from clients.chroma_client import get_collection
from graph.general_query import get_general_nodes_context
from graph.related_entities import get_related_entities
from graph.specific_nodes import get_specific_nodes_context
from graph.top_nodes import get_top_nodes_context
from src.core.config import (
    CODEBERT_MODEL_NAME,
    NODE_CONTEXT_HISTORY,
    projects,
)
from src.core.intent_analyzer import get_intent_analyzer
from src.core.models import (
    ConversationHistory,
    PerformanceMetrics,
    PrompRequest,
    SimpleRAGResponse,
)

load_dotenv()
from src.core.prompt import build_prompt
from testing.token_counter import count_tokens



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORY_LIMIT = 5
BASE_DIR = os.path.abspath(projects)

timeout: httpx.Timeout = httpx.Timeout(200.0)
client: httpx.AsyncClient = httpx.AsyncClient(timeout=timeout)
conversation_history: ConversationHistory = ConversationHistory(max_history_pairs=HISTORY_LIMIT)



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
        logger.info("Graph model zaladowany")
        _get_cached_model()
        logger.info("CodeBERT model zaladowany")
        warum_time = time.time() - start_time
        logger.info(f"RAG models zaladowane w {warum_time:.2f}s")
    except Exception as e:
        logger.error(f"Model warmup failed: {e}")


def log_metrics(metrics: PerformanceMetrics) -> None:
    """
    Logs performance metrics for a single endpoint call.
    Includes total, RAG, and LLM times, as well as context and response lengths.
    """
    logger.info(
        f"Performance [{metrics.endpoint}]: "
        f"total={metrics.total_time:.3f}s, "
        f"rag={metrics.rag_time:.3f}s, "
        f"llm={metrics.llm_time:.3f}s, "
        f"context_len={metrics.context_length}, "
        f"response_len={metrics.response_length}"
    )


@app.post("/ask", response_model=SimpleRAGResponse)
async def ask(req: PrompRequest) -> SimpleRAGResponse:
    """
    Handles the `/ask` endpoint that sends a question and context to the LLM.

    Builds a prompt from the request, queries the model, measures response times,
    logs performance metrics, and returns the generated answer.

    Args:
        req (PrompRequest): Request object containing the `question` and `context`.

    Returns:
        SimpleRAGResponse: The model-generated answer and total processing time.

    Raises:
        HTTPException: If validation fails or an unexpected error occurs.
    """
    start_time = time.time()
    try:
        prompt = f"Context: {req.context}\n\nQuestion: {req.question}\nAnswer:"
        llm_start = time.time()
        answer = await get_llm_response(prompt)
        llm_time = time.time() - llm_start
        total_time = time.time() - start_time
        response = SimpleRAGResponse(answer=answer, processing_time=total_time)
        metrics = PerformanceMetrics(
            endpoint="/ask",
            total_time=total_time,
            llm_time=llm_time,
            context_length=len(req.context),
            response_length=len(answer),
        )
        log_metrics(metrics)
        return response
    except ValidationError as e:
        logger.error(f"Validation error in /ask: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Validation error: {str(e)}"
        )
    except Exception as exc:
        logger.error(f"Unexpected error in /ask: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(exc)}",
        )


async def load_code(file_path: str) -> str:
    """
    Loads and returns the content of a code file.

    Checks if the file exists and is non-empty before returning its contents.
    Logs and raises appropriate HTTP exceptions on failure.

    Args:
        file_path (str): Path to the file to be read.

    Returns:
        str: The text content of the specified file.

    Raises:
        HTTPException: If the file is not found, empty, or cannot be read.
    """
    if not os.path.exists(file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    try:
        with open(file_path, "r") as f:
            content = f.read()
        if not content.strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty")
        return content
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to read file"
        )


@app.post("/ask_code", response_model=SimpleRAGResponse)
async def ask_code(file_path: str, question: str) -> SimpleRAGResponse:
    """
    Handles the `/ask_code` endpoint that queries the LLM about a code file.

    Loads the code from the given file, constructs a prompt with the provided
    question, sends it to the model via the `/ask` endpoint, and returns the response.

    Args:
        file_path (str): Path to the source code file to analyze.
        question (str): The question to ask about the code.

    Returns:
        SimpleRAGResponse: The model-generated answer and total processing time.

    Raises:
        HTTPException: Propagates any exceptions from file loading or model interaction.
    """
    start_time = time.time()
    try:
        code = await load_code(file_path)
        prompt_request = PrompRequest(question=question, context=code)
        response = await ask(prompt_request)
        response.processing_time = time.time() - start_time
        return response
    except Exception as exc:
        logger.error(f"Error in /ask_code: {exc}")
        raise


class AskRequest(BaseModel):
    """
    Request model for basic question queries.

    Attributes:
        question (str): The user's input question.
        params (dict): The agent's input parameters specific for each endpoint.
    """

    question: str
    params: dict | None = None


class AskResponse(BaseModel):
    """
    Response model for basic RAG question results.

    Attributes:
        status (str): Request status, defaults to "success".
        context (str): Retrieved or generated context relevant to the question.
        matches (int): Number of context matches found, defaults to 0.
    """

    status: str = "success"
    context: str
    matches: int = 0


@app.post("/ask_specific_nodes")
async def ask_specific_nodes(req: AskRequest):
    return await retrieve_context(req.question, get_specific_nodes_context, req.params)


@app.post("/ask_top_nodes")
async def ask_top_nodes(req: AskRequest):
    return await retrieve_context(req.question, get_top_nodes_context, req.params)


@app.post("/ask_general_question")
async def ask_general_question(req: AskRequest):
    return await retrieve_context(req.question, get_general_nodes_context, req.params)


@app.post("/list_related_entities")
async def list_related_entities(req: AskRequest):
    return await retrieve_context(req.question, get_related_entities, req.params)


async def retrieve_context(question: str, node_func, params: dict):
    """
    Handles the all the endpoints for quick RAG-style context retrieval.

    Uses a similarity model to find nodes relevant to the given question,
    returning the matched context and number of matches.

    Args:
        question (str): User question
        node_func (function): Function used to retrieve nodes relevant to the given question

    Returns:
        dict: A dictionary with the fields:
            - status (str): "success" or "error"
            - context (str): Retrieved context or error message
            - matches (int): Number of matching nodes found
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
            timeout=60.0,
        )
        rag_time = time.time() - rag_start_time
        logger.info(f"RAG processing: {rag_time:.3f}s, found {len(matches)} matches")

        history_start_time = time.time()
        try:
            with open(NODE_CONTEXT_HISTORY, "r", encoding="utf-8") as f:
                history_data = json.load(f)
            conversation_history.messages.clear()
            for msg in history_data:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    conversation_history.add_message(msg["role"], msg["content"])
        except FileNotFoundError:
            logger.info("No conversation history found")
            conversation_history.messages.clear()
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Invalid conversation history format: {e}")
            conversation_history.messages.clear()
        history_load_time = time.time() - history_start_time
        logger.info(
            f"History loaded: {history_load_time:.3f}s, "
            f"{len(conversation_history.messages)} messages"
        )

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
        # Zmienić liczenie na codebert
        context_tokens = count_tokens(context, "llama")
        prompt_tokens = count_tokens(prompt, "llama")
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
