import json
import os
import re
from typing import Any, Dict, List, Tuple

from loguru import logger
from sentence_transformers import SentenceTransformer

from src.clients.chroma_client import (
    default_collection_name,
    get_chroma_client,
    get_collection,
)
from src.clients.llm_client import call_llm
from src.core.config import (
    CODEBERT_MODEL_NAME,
    default_chroma_path,
    default_classifier_embeddings_path,
    default_classifier_model,
)
from src.core.intent_analyzer import classify_question, get_intent_analyzer
from src.graph.generate_embeddings_graph import generate_embeddings_graph

default_top_k = 7


def load_classifier_embeddings(path: str = None) -> dict:
    """
    Loads serialized classifier embeddings from disk.

    Args:
        path (str, optional): Path to the embeddings JSON file.
            Defaults to `default_classifier_embeddings_path`.

    Returns:
        dict: Parsed embeddings payload.

    Raises:
        FileNotFoundError: If the embeddings file does not exist.
    """
    embeddings_path = path or default_classifier_embeddings_path
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError("Classifier embeddings not found")
    with open(embeddings_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_classifier_model(model_name: str = None) -> SentenceTransformer:
    """
    Initializes and returns the sentence-transformer classifier model.

    Args:
        model_name (str, optional): Hugging Face model name.
            Defaults to `default_classifier_model`.

    Returns:
        SentenceTransformer: Loaded sentence-transformer model.
    """
    model_name = model_name or default_classifier_model
    return SentenceTransformer(model_name)


try:
    classifier_embeddings = load_classifier_embeddings()
    classifier_model = get_classifier_model()
except Exception as e:
    logger.error(f"Error with loading classifier components: {e}")
    classifier_embeddings = {}
    classifier_model = None

chroma_client = get_chroma_client(storage_path=default_chroma_path)


async def extract_key_value_pairs_simple(question: str) -> List[Tuple[str, str]]:
    """
    Extracts (key, value) pairs from a question using LLM.

    Args:
        question (str): User question in natural language.

    Returns:
        List[Tuple[str, str]]: Unique (key, value) pairs such as
        ('class', 'orderservice') or ('method', 'findById').
    """
    question_lower = question.lower()
    classification_prompt = f"""
    User question: "{question_lower}"
    Your task:
    1. Extract pairs of (Node Type, Node Name) from the question
    2. Node Type must be one of: ["CLASS", "METHOD", "VARIABLE", "CONSTRUCTOR", "VALUE"],
     don't pick other option. It is INVALID.
    3. For tests assume that node type is: CLASS
    4. Return ONLY a valid JSON array of objects with keys "type" and "name"

    Example output format:
    [
      {{"type": "CLASS", "name": "User"}},
      {{"type": "METHOD", "name": "findById"}}
    ]
    - Return ONLY valid JSON, no comments, no explanations, no markdown
    - Always return a JSON array, even if there's only 1 element
    - If no pairs found, return empty array: []

    Return your JSON without ```json only array of pairs
    """

    answer = await call_llm(classification_prompt)
    logger.debug(f"LLM extracted pairs: {answer}")
    try:
        data = json.loads(answer)
        logger.debug(f"Data: {data}")
        pairs = [(item["type"].upper(), item["name"]) for item in data]
    except Exception as e:
        logger.error(f"Failed to parse json asner: {e}")
        pairs = []
    return pairs


def preprocess_question(q: str) -> str:
    """
    Normalizes a question string to aid lightweight classification.

    Replaces concrete mentions like "method foo" with generic tokens
    ("method", "function", "class", "variable"), collapses whitespace,
    and lowercases the text.

    Args:
        q (str): Raw question text.

    Returns:
        str: Normalized, lowercased question string.
    """
    q = re.sub(r"\bmethod\s+\w+\b", "method", q, flags=re.IGNORECASE)
    q = re.sub(r"\bfunction\s+\w+\b", "function", q, flags=re.IGNORECASE)
    q = re.sub(r"\bclass\s+\w+\b", "class", q, flags=re.IGNORECASE)
    q = re.sub(r"\bvariable\s+\w+\b", "variable", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip()
    return q.lower()


def similar_node(
    question: str,
    model_name: str = CODEBERT_MODEL_NAME,
    collection_name: str = default_collection_name,
    top_k: int = default_top_k,
) -> Tuple[List[Tuple[float, Dict[str, Any]]], str]:
    """
    Retrieves the most similar code nodes and builds a context string.

    1) Extracts (key, value) pairs from the question.
    2) Generates embeddings (CodeBERT) for the query variants.
    3) Queries a Chroma collection for top-k similar nodes per query.
    4) Expands results with related neighbors (importance-ranked) based on
       stored metadata and detected intent category.
    5) Returns unique top nodes and a concatenated textual context.

    Args:
        question (str): User question to search for.
        model_name (str, optional): Embedding model name. Defaults to `CODEBERT_MODEL_NAME`.
        collection_name (str, optional): Chroma collection to query.
            Defaults to `default_collection_name`.
        top_k (int, optional): Number of results to retrieve per query embedding.
            Defaults to `default_top_k`.

    Returns:
        Tuple[List[Tuple[float, Dict[str, Any]]], str]:
            - List of (score, node) tuples sorted by similarity (unique nodes).
            - Context string composed of code snippets (or fallback text).

    Notes:
        - Similarity score is computed as `1 - distance` from Chroma results.
        - When category is "general" and no hits are found, falls back to
          top documents by importance.

    Raises:
        This function handles most exceptions internally (logging warnings/errors)
        and returns best-effort results. It does not raise on Chroma/query errors.
    """
    collection = get_collection("scg_embeddings")
    pairs = extract_key_value_pairs_simple(question)
    embeddings_input = []
    for key, value in pairs:
        embeddings_input.append(f"{key} {value}" if key else value)
    if not embeddings_input:
        embeddings_input = [question]
    query_embeddings = generate_embeddings_graph(embeddings_input, model_name)
    results = []
    for query_emb in query_embeddings:
        try:
            query_result = collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=top_k,
                include=["embeddings", "metadatas", "documents", "distances"],
            )
            for i in range(len(query_result["ids"][0])):
                score = 1 - query_result["distances"][0][i]
                node_id = query_result["ids"][0][i]
                metadata = query_result["metadatas"][0][i]
                code = query_result["documents"][0][i]
                results.append((score, {"node": node_id, "metadata": metadata, "code": code}))
        except Exception as e:
            logger.error(f"Error querying collection: {e}")

    seen = set()
    unique_results = []
    for score, node in sorted(results, key=lambda x: -x[0]):
        if node["node"] not in seen:
            unique_results.append((score, node))
            seen.add(node["node"])
        if len(unique_results) >= len(embeddings_input) * top_k:
            break
    top_nodes = unique_results[: len(embeddings_input)]
    top_k_codes = [node["code"] for _, node in top_nodes if node["code"]]
    try:
        analyzer = get_intent_analyzer()
        analysis = analyzer.enhanced_classify_question(question)
        category = analysis.primary_intent.value
    except Exception as e:
        logger.warning(f"Fallback to basic classification: {e}")
        category = classify_question(preprocess_question(question))

    max_neighbors = {
        "general": 5,
        "usage": 3,
        "definition": 2,
        "implementation": 3,
        "testing": 4,
        "exception": 3,
        "top": 1,
    }.get(category, 2)
    logger.debug(f"Category: {category}")

    all_neighbors_ids = set()
    for _, node in top_nodes:
        neighbors = node["metadata"].get("related_entities", [])
        if isinstance(neighbors, str):
            try:
                neighbors = json.loads(neighbors)
            except json.JSONDecodeError:
                neighbors = []
        all_neighbors_ids.update(neighbors)

    neighbor_codes = []
    if all_neighbors_ids:
        try:
            neighbor_nodes = collection.get(
                ids=list(all_neighbors_ids), include=["documents", "metadatas"]
            )

            neighbors_with_scores = []
            for i in range(len(neighbor_nodes["ids"])):
                nid = neighbor_nodes["ids"][i]
                meta = neighbor_nodes["metadatas"][i]
                doc = neighbor_nodes["documents"][i]

                if doc:
                    score = meta.get("combined", 0.0)
                    neighbors_with_scores.append((score, nid, doc))

            sorted_neighbors = sorted(neighbors_with_scores, key=lambda x: -x[0])
            neighbor_codes = [doc for _, _, doc in sorted_neighbors[:max_neighbors]]
        except Exception as e:
            logger.error(f"Error getting neighbors: {e}")

    all_codes = []
    seen_codes = set()
    for code in top_k_codes + neighbor_codes:
        if code and code not in seen_codes and not code.startswith("<"):
            all_codes.append(code)
            seen_codes.add(code)

    full_context = "\n\n".join(all_codes)

    if not all_codes and category == "general":
        try:
            all_nodes = collection.get(include=["documents", "metadatas", "ids"])
            importance_scores = []
            for i in range(len(all_nodes["ids"])):
                doc = all_nodes["documents"][i]
                meta = all_nodes["metadatas"][i]
                nid = all_nodes["ids"][i]
                score = (
                    meta.get("importance", {}).get("combined", 0.0)
                    if isinstance(meta.get("importance"), dict)
                    else meta.get("combined", 0.0)
                )
                if doc:
                    importance_scores.append((score, nid, doc))
            sorted_by_importance = sorted(importance_scores, key=lambda x: -x[0])
            fallback_docs = [doc for _, _, doc in sorted_by_importance[:5]]
            full_context = "\n\n".join(fallback_docs)
        except Exception as e:
            logger.error(f"Error retrieving fallback for general question: {e}")
            full_context = "<NO CONTEXT FOUND>"
    return top_nodes, full_context or "<NO CONTEXT FOUND>"
