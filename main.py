import json
import logging
import os
from typing import List, Dict, Any
import time
import httpx
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from models import PrompRequest, SimpleRAGResponse, NodeRAGResponse, PerformanceMetrics, ConversationHistory
from graph.similar_node_optimization import similar_node_fast
from prompt import build_intent_aware_prompt, post_process_answer, analyze_user_intent

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

OLLAMA_API_URL = "http://localhost:11434/api/generate"
# MODEL_NAME = "deepseek-coder:6.7b"
# MODEL_NAME = "codellama:7b-instruct"
MODEL_NAME = "llama3.1:8b"
NODE_EMBEDDINGS = "embeddings/node_embedding.json"
NODE_CONTEXT_HISTORY = "embeddings/node_context_history.json"
HISTORY_LIMIT = 5
CODEBERT_MODEL_NAME = "microsoft/codebert-base"
BASE_DIR = os.path.abspath("projects")

timeout: httpx.Timeout = httpx.Timeout(200.0)
client: httpx.AsyncClient = httpx.AsyncClient(timeout=timeout)
conversation_history: ConversationHistory = ConversationHistory(max_history_pairs=HISTORY_LIMIT)


def warm_up_models() -> None:
    logger.info("Warming up models...")
    start_time = time.time()

    try:
        from graph.similar_node_optimization import get_graph_model
        from rag_optimization import _get_cached_model

        get_graph_model()
        logger.info("Graph model zaladowany")

        _get_cached_model()
        logger.info("CodeBERT model zaladowany")

        warum_time = time.time() - start_time
        logger.info(f"RAG models zaladowane w {warum_time:.2f}s")

    except Exception as e:
        logger.error(f"Model warmup failed: {e}")


def log_performance_metrics(metrics: PerformanceMetrics) -> None:
    logger.info(
        f"Performance [{metrics.endpoint}]: "
        f"total={metrics.total_time:.3f}s, "
        f"rag={metrics.rag_time:.3f}s, "
        f"llm={metrics.llm_time:.3f}s, "
        f"context_len={metrics.context_length}, "
        f"response_len={metrics.response_length}"
    )


async def get_llm_response(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
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
        log_performance_metrics(metrics)
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
    total_start_time = time.time()
    try:
        intent_start_time = time.time()
        user_intent = analyze_user_intent(req.question)
        intent_time = time.time() - intent_start_time
        logger.info(f"User intent analysis: {intent_time:.3f}s - {user_intent}")
        rag_start_time = time.time()
        matches, context = similar_node_fast(req.question, model_name=CODEBERT_MODEL_NAME)
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
        prompt = build_intent_aware_prompt(
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

        from graph.retriver import enhanced_classify_question
        analysis = enhanced_classify_question(req.question)
        category = analysis.get("category", "general")

        response = NodeRAGResponse(
            answer=processed_answer,
            used_context=context,
            processing_time=total_time,
            question_category=category
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
        log_performance_metrics(metrics)
        return response

    except Exception as exc:
        logger.error(f"Error in RAG Node: {str(exc)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in RAG Node: {str(exc)}"
        )



@app.get("/files", response_model=List[str])
def list_files(path: str = "") -> List[str]:
    target_dir = os.path.abspath(os.path.join(BASE_DIR, path))
    if not target_dir.startswith(BASE_DIR):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    try:
        if not os.path.exists(target_dir):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Folder not found"
            )
        files = os.listdir(target_dir)
        return sorted(files)
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied"
        )


@app.get("/file")
def get_file(path: str = Query(..., description="Relative path to the file")) -> dict:
    file_path = os.path.abspath(os.path.join(BASE_DIR, path))
    if not file_path.startswith(BASE_DIR):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    if not os.path.isfile(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {
            "content": content,
            "file_path": path,
            "size": len(content),
            "timestamp": time.time()
        }
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is not text or has unsupported encoding"
        )
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.on_event("startup")
async def startup_event():
    logger.info("Starting RAG application...")
    warm_up_models()

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
    logger.info("Shutting down RAG application...")
    await client.aclose()
    logger.info("HTTP client closed")