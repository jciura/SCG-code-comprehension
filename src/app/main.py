import asyncio
import json
import logging
import os
import time
from typing import List, Dict

import httpx
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError, BaseModel

from src.graph.similar_node_optimization import similar_node_fast
from src.core.intent_analyzer import get_intent_analyzer

from src.core.models import PrompRequest, SimpleRAGResponse, NodeRAGResponse, PerformanceMetrics, ConversationHistory
from src.core.prompt import build_prompt, post_process_answer
from src.core.config import MODEL_NAME, OLLAMA_API_URL, NODE_CONTEXT_HISTORY, CODEBERT_MODEL_NAME, projects, ground_truth, metrics_path

from src.core.evaluator import RAGEvaluator

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

evaluator: RAGEvaluator = None


def warm_up_models() -> None:
    """
        Preloads key models to reduce startup latency.
        Logs progress, total warm-up time, and any errors encountered.
        Does not raise exceptions.
    """
    logger.info("Warming up models...")
    start_time = time.time()
    try:
        from src.graph.similar_node_optimization import get_graph_model
        from src.core.rag_optimization import _get_cached_model
        get_graph_model()
        logger.info("Graph model zaladowany")
        _get_cached_model()
        logger.info("CodeBERT model zaladowany")
        warum_time = time.time() - start_time
        logger.info(f"RAG models zaladowane w {warum_time:.2f}s")
    except Exception as e:
        logger.error(f"Model warmup failed: {e}")


def load_evaluator() -> None:
    """
        Loads the RAG evaluator if the ground truth file exists.

        Initializes a global `evaluator` using RAGEvaluator with LLM judging.
        Logs success or failure; sets `evaluator` to None on error or missing file.
    """
    global evaluator
    ground_truth_path = ground_truth
    if os.path.exists(ground_truth_path):
        try:
            evaluator = RAGEvaluator(test_suite_path=ground_truth_path,use_llm_judge=True)
        except Exception as e:
            logger.error(f"Failed to load evaluator: {e}")
            evaluator = None
    else:
        logger.info(f"Ground truth file not found at {ground_truth_path}")
        evaluator = None


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
        f"response_len={metrics.response_length}")




async def evaluate_if_needed(question: str, answer: str, context: str) -> Dict:
    """
        Evaluates a model's answer against ground truth if available.
        Finds the matching question in the evaluator's test suite and computes
        comprehensive RAG metrics (RAGAS, context recall, faithfulness).
        Logs results and returns evaluation data or None on failure.

        Args:
            question (str): The user or test question to evaluate.
            answer (str): The model-generated answer.
            context (str): Retrieved context(s).

        Returns:
            Dict | None: A dictionary containing question ID, category, and
            computed metrics, or None if evaluation is skipped or fails.
    """
    if evaluator is None:
        return None
    matching_question = None
    for test_question in evaluator.test_suite.questions:
        if test_question.question.lower().strip() == question.lower().strip():
            matching_question = test_question
            break
    if matching_question is None:
        logger.debug(f"Question not found in ground truth")
        return None
    retrieved_contexts = []
    for ctx in context.split('##'):
        ctx = ctx.strip()
        if ctx:
            retrieved_contexts.append(ctx)
    try:
        metrics_report = evaluator.rag_metrics.full_evaluation(
            question=question,
            answer=answer,
            retrieved_contexts=retrieved_contexts,
            ground_truth_answer=matching_question.correct_answer,
            ground_truth_contexts=matching_question.ground_truth_contexts,
            key_entities=matching_question.key_entities,
            key_facts=matching_question.key_facts)
        return {
            "question_id": matching_question.id,
            "category": matching_question.category,
            "metrics": metrics_report}
    except Exception as e:
        logger.error(f"Evaluation failed for question {matching_question.id}: {e}")
        logger.exception(e)
        return None


async def get_llm_response(prompt: str) -> str:
    """
        Sends a prompt to the Ollama API and returns the model's response.

        Builds and sends a JSON payload with model parameters, validates the response,
        and handles HTTP and request-related errors gracefully.

        Args:
            prompt (str): The input text prompt to send to the language model.

        Returns:
            str: The generated text response from the model.

        Raises:
            HTTPException: If the Ollama API returns an error, invalid format,
                or an empty response.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            "repeat_penalty": 1.1
        }}
    try:
        response = await client.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()

        if "response" not in result:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Invalid response format from Ollama"
            )

        answer = result["response"]
        if not answer or not answer.strip():
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Empty response from Ollama"
            )
        return answer.strip()
    except httpx.HTTPStatusError as exc:
        error_detail = f"Ollama API error {exc.response.status_code}"
        try:
            error_details = exc.response.json()
            error_detail += f": {error_details}"
        except Exception:
            error_detail += f": {exc.response.text}"
        logger.error(error_detail)
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=error_detail
        )
    except httpx.RequestError as exc:
        logger.error(f"Request error while calling Ollama API: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Request error: {str(exc)}"
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
        response = SimpleRAGResponse(
            answer=answer,
            processing_time=total_time
        )
        metrics = PerformanceMetrics(
            endpoint="/ask",
            total_time=total_time,
            llm_time=llm_time,
            context_length=len(req.context),
            response_length=len(answer)
        )
        log_metrics(metrics)
        return response
    except ValidationError as e:
        logger.error(f"Validation error in /ask: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as exc:
        logger.error(f"Unexpected error in /ask: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(exc)}"
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    try:
        with open(file_path, "r") as f:
            content = f.read()
        if not content.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty"
            )
        return content
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read file"
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


@app.post("/ask_rag_node", response_model=NodeRAGResponse)
async def ask_rag_node(req: PrompRequest) -> NodeRAGResponse:
    """
        Handles the `/ask_rag_node` endpoint for intent-aware RAG queries on code nodes.

        Performs user intent classification, retrieves semantically similar nodes,
        builds a context-aware prompt, queries the LLM, post-processes the response,
        and evaluates results against ground truth when available.
        Also manages conversation history persistence and logs detailed performance metrics.

        Args:
            req (PrompRequest): Request object containing the user's question and optional context.

        Returns:
            NodeRAGResponse: The model's processed answer, used context, total processing time,
            and detected question category.

        Raises:
            HTTPException: If any step in the RAG or LLM pipeline fails (e.g., intent analysis,
            model call, history load/save, or evaluation).
    """
    total_start_time = time.time()
    try:
        analyzer = get_intent_analyzer()
        intent_start_time = time.time()
        user_intent_analysis = analyzer.enhanced_classify_question(req.question)
        user_intent = {
            "primary_intent": user_intent_analysis.primary_intent.value,
            "secondary_intents": user_intent_analysis.secondary_intents,
            "requires_examples": user_intent_analysis.requires_examples,
            "requires_usage_info": user_intent_analysis.requires_usage_info,
            "requires_implementation_details": user_intent_analysis.requires_implementation_details,
        }
        intent_time = time.time() - intent_start_time
        logger.info(f"User intent analysis: {intent_time:.3f}s - {user_intent}")
        rag_start_time = time.time()
        matches, context = await similar_node_fast(req.question, model_name=CODEBERT_MODEL_NAME)
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
        logger.info(f"History loaded: {history_load_time:.3f}s, {len(conversation_history.messages)} messages")

        prompt_start_time = time.time()
        prompt = build_prompt(
            question=req.question,
            context=context,
            intent=user_intent,
            conversation_history=conversation_history
        )
        prompt_time = time.time() - prompt_start_time
        logger.info(f"Intent-aware prompt built: {prompt_time:.3f}s, length: {len(prompt)} chars")

        if logger.level <= logging.DEBUG:
            logger.debug("Generated prompt:")
            logger.debug(prompt)

        llm_start_time = time.time()
        answer = await get_llm_response(prompt)
        llm_time = time.time() - llm_start_time
        logger.info(f"LLM response: {llm_time:.3f}s, length: {len(answer)} chars")

        if logger.level <= logging.DEBUG:
            logger.debug("LLM answer:")
            logger.debug(answer)

        processed_answer = post_process_answer(answer, user_intent)
        evaluation_result = await evaluate_if_needed(req.question, processed_answer, context)
        if evaluation_result:
            try:
                eval_log_path = metrics_path
                os.makedirs(os.path.dirname(eval_log_path), exist_ok=True)
                with open(eval_log_path, 'a', encoding='utf-8') as f:
                    eval_entry = {
                        "timestamp": time.time(),
                        "question": req.question,
                        "answer": processed_answer,
                        "question_id": evaluation_result["question_id"],
                        "category": evaluation_result["category"],
                        "ragas_score": evaluation_result["metrics"]["ragas"]["ragas_score"],
                        "full_metrics": evaluation_result["metrics"]}
                    f.write(json.dumps(eval_entry, ensure_ascii=False, indent=2) + '\n\n')
                    logger.warning(f"EVALUATION RESULT: {evaluation_result}")
            except Exception as e:
                logger.warning(f"Failed to save evaluation result: {e}")
        history_save_start_time = time.time()
        conversation_history.add_message("user", req.question)
        conversation_history.add_message("assistant", processed_answer)

        try:
            history_data = [
                {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp}
                for msg in conversation_history.messages
            ]

            with open(NODE_CONTEXT_HISTORY, "w", encoding="utf-8") as f:
                json.dump(history_data, f, ensure_ascii=False, indent=4)

        except Exception as e:
            logger.error(f"Failed to save conversation history: {e}")

        history_save_time = time.time() - history_save_start_time
        logger.info(f"History saved: {history_save_time:.3f}s")
        total_time = time.time() - total_start_time

        response = NodeRAGResponse(
            answer=processed_answer,
            used_context=context,
            processing_time=total_time,
            question_category=user_intent_analysis.primary_intent.value
        )

        metrics = PerformanceMetrics(
            endpoint="/ask_rag_node",
            total_time=total_time,
            rag_time=rag_time,
            llm_time=llm_time,
            history_load_time=history_load_time,
            history_save_time=history_save_time,
            context_length=len(context),
            response_length=len(processed_answer)
        )
        log_metrics(metrics)
        return response

    except Exception as exc:
        logger.error(f"Error in RAG Node: {str(exc)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in RAG Node: {str(exc)}"
        )


class AskRequest(BaseModel):
    """
        Request model for basic question queries.

        Attributes:
            question (str): The user's input question.
    """
    question: str


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


@app.post("/ask_junie")
async def ask_junie(req: AskRequest):
    """
        Handles the `/ask_junie` endpoint for quick RAG-style context retrieval.

        Uses a similarity model to find nodes relevant to the given question,
        returning the matched context and number of matches.

        Args:
            req (AskRequest): Request containing the user's question.

        Returns:
            dict: A dictionary with the fields:
                - status (str): "success" or "error"
                - context (str): Retrieved context or error message
                - matches (int): Number of matching nodes found
        """
    total_start_time = time.time()
    logger.info(f"Received question: {req.question}")

    try:
        analyzer = get_intent_analyzer()
        intent_start_time = time.time()
        user_intent_analysis = analyzer.enhanced_classify_question(req.question)
        user_intent = {
            "primary_intent": user_intent_analysis.primary_intent.value,
            "secondary_intents": user_intent_analysis.secondary_intents,
            "requires_examples": user_intent_analysis.requires_examples,
            "requires_usage_info": user_intent_analysis.requires_usage_info,
            "requires_implementation_details": user_intent_analysis.requires_implementation_details,
        }
        intent_time = time.time() - intent_start_time
        logger.info(f"User intent analysis: {intent_time:.3f}s - {user_intent}")
        rag_start_time = time.time()
        matches, context = await asyncio.wait_for(
            similar_node_fast(req.question, model_name=CODEBERT_MODEL_NAME),
            timeout=30.0
        )
        rag_time = time.time() - rag_start_time
        logger.info(f"RAG processing: {rag_time:.3f}s, found {len(matches)} matches")

        prompt_start_time = time.time()
        prompt = build_prompt(
            question=req.question,
            context=context,
            intent=user_intent,
            conversation_history=conversation_history
        )
        prompt_time = time.time() - prompt_start_time
        logger.info(f"Intent-aware prompt built: {prompt_time:.3f}s, length: {len(prompt)} chars")

        if logger.level <= logging.DEBUG:
            logger.debug("Generated prompt:")
            logger.debug(prompt)


        if not isinstance(context, str):
            context = str(context)

        return {
            "status": "success",
            "context": context,
            "matches": len(matches) if matches else 0
        }

    except Exception as e:
        return {
            "status": "error",
            "context": f"Error: {str(e)}",
            "matches": 0
        }


@app.on_event("startup")
async def startup_event():
    """
        Application startup event handler.

        Initializes models, evaluator, and verifies the Ollama server connection.
        Keeps the selected model in memory and performs a warm-up request to
        reduce first-call latency.

        Logs the status of each initialization step and reports any failures.
    """
    logger.info("Starting RAG application...")
    warm_up_models()

    load_evaluator()

    try:
        logger.info("Checking Ollama server connection...")
        health_response = await client.get("http://localhost:11434/api/tags")
        health_response.raise_for_status()
        logger.info("Ollama server is running")
    except httpx.ConnectError:
        logger.error("Cannot connect to Ollama server")
        return
    except Exception as e:
        logger.warning(f"Health check failed: {e}")

    try:
        keep_alive_payload = {
            "model": MODEL_NAME,
            "prompt": "",
            "keep_alive": -1
        }
        await client.post(OLLAMA_API_URL, json=keep_alive_payload)
        logger.info(f"Model {MODEL_NAME} configured to stay in memory permanently")
    except Exception as e:
        logger.warning(f"Failed to configure model persistence: {e}")

    try:
        start_time = time.time()
        warmup_payload = {
            "model": MODEL_NAME,
            "prompt": "This is a warmup.",
            "stream": False,
            "options": {"num_predict": 5}
        }

        response = await client.post(OLLAMA_API_URL, json=warmup_payload)
        response.raise_for_status()

        warmup_time = time.time() - start_time
        logger.info(f"Model {MODEL_NAME} warmed up successfully in {warmup_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Model warmup failed: {e}")

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