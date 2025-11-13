import json
import os
import re
from loguru import logger
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer
from src.clients.chroma_client import get_chroma_client, get_collection, default_collection_name
from src.graph.generate_embeddings_graph import generate_embeddings_graph
from src.core.intent_analyzer import get_intent_analyzer, classify_question
from src.core.config import default_classifier_embeddings_path, default_chroma_path, default_classifier_model, CODEBERT_MODEL_NAME

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
        raise FileNotFoundError(f"Classifier embeddings not found")
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


def extract_key_value_pairs_simple(question: str) -> List[Tuple[str, str]]:
    """
        Extracts (key, value) pairs from a question.

        Looks for signals like `class`, `method`, Java-style names, camelCase tokens,
        method prefixes (find/get/set/...), and compound `Class.method` patterns.

        Args:
            question (str): User question in natural language.

        Returns:
            List[Tuple[str, str]]: Unique (key, value) pairs such as
            ('class', 'orderservice') or ('method', 'findById').
        """
    pairs = []
    question_lower = question.lower()
    words = question_lower.split()
    key_terms = {"class", "method", "function", "variable", "property"}
    for i, word in enumerate(words):
        if word in key_terms:
            if i + 1 < len(words):
                next_word = words[i + 1]
                clean_name = next_word.strip("'\"().,!?")
                if clean_name and len(clean_name) > 1:
                    pairs.append((word, clean_name))
            elif i > 0:
                prev_word = words[i - 1]
                clean_name = prev_word.strip("'\"().,!?")
                if clean_name and len(clean_name) > 1:
                    pairs.append((word, clean_name))
    java_class_pattern = r'\b(\w+(?:service|controller|repository|dto|entity|exception))\b'
    java_matches = re.findall(java_class_pattern, question_lower)
    for match in java_matches:
        pairs.append(('class', match))
    for word in re.findall(r'\b[A-Z][a-zA-Z]+\b', question):
        word_lower = word.lower()
        if (word_lower.endswith(('service', 'controller', 'repository', 'dto', 'entity', 'exception')) or
                len(word) > 8):
            pairs.append(('class', word_lower))
    camel_case_pattern = r'\b([a-z]+[A-Z][a-zA-Z0-9]*)\b'
    camel_matches = re.findall(camel_case_pattern, question)
    for match in camel_matches:
        if len(match) > 10:
            pairs.append(('method', match.lower()))
            logger.debug(f"Extracted camelCase method: {match}")
        elif len(match) > 6:
            pairs.append(('method', match.lower()))
    method_with_parens = re.findall(r'\b(\w+)\s*\(\s*\)', question_lower)
    for method_name in method_with_parens:
        pairs.append(('method', method_name))
        logger.debug(f"Extracted method with parens: {method_name}")
    method_prefixes = ['find', 'get', 'set', 'create', 'update', 'delete', 'add', 'remove',
                       'enroll', 'unenroll', 'save', 'fetch', 'load', 'check', 'validate']
    for word in words:
        clean_word = word.strip("'\"().,!?")
        for prefix in method_prefixes:
            if clean_word.startswith(prefix) and len(clean_word) > len(prefix) + 2:
                pairs.append(('method', clean_word))
                logger.debug(f"Extracted method with prefix: {clean_word}")
                break
    pattern = r'(\w+)\s+method\s+in\s+(\w+)\s+class'
    matches = re.findall(pattern, question_lower)
    for method, class_name in matches:
        pairs.append(('method', method))
        pairs.append(('class', class_name))
    compound_pattern = r'([A-Z][a-zA-Z]+)\.([a-z][a-zA-Z]+)'
    compound_matches = re.findall(compound_pattern, question)
    for class_name, method_name in compound_matches:
        pairs.append(('class', class_name.lower()))
        pairs.append(('method', method_name.lower()))
        logger.debug(f"Extracted compound: {class_name}.{method_name}")
    long_words = re.findall(r'\b([a-zA-Z]{15,})\b', question)
    for word in long_words:
        if any(c.isupper() for c in word[1:]):
            if word[0].isupper():
                pairs.append(('class', word.lower()))
            else:
                pairs.append(('method', word.lower()))
            logger.debug(f"Extracted long word: {word}")
    seen = set()
    unique_pairs = []
    for pair in pairs:
        if pair not in seen:
            unique_pairs.append(pair)
            seen.add(pair)
    if unique_pairs:
        logger.info(f"Extracted {len(unique_pairs)} unique pairs from question")
        for key, value in unique_pairs[:5]:
            logger.debug(f"  :{key}: {value}")
    return unique_pairs


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
    q = re.sub(r'\bmethod\s+\w+\b', 'method', q, flags=re.IGNORECASE)
    q = re.sub(r'\bfunction\s+\w+\b', 'function', q, flags=re.IGNORECASE)
    q = re.sub(r'\bclass\s+\w+\b', 'class', q, flags=re.IGNORECASE)
    q = re.sub(r'\bvariable\s+\w+\b', 'variable', q, flags=re.IGNORECASE)
    q = re.sub(r'\s+', ' ', q).strip()
    return q.lower()


def similar_node(question: str, model_name: str = CODEBERT_MODEL_NAME, collection_name: str = default_collection_name, top_k: int = default_top_k) -> Tuple[List[Tuple[float, Dict[str, Any]]], str]:
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
                include=["embeddings", "metadatas", "documents", "distances"])
            for i in range(len(query_result["ids"][0])):
                score = 1 - query_result["distances"][0][i]
                node_id = query_result["ids"][0][i]
                metadata = query_result["metadatas"][0][i]
                code = query_result["documents"][0][i]
                results.append((score, {
                    "node": node_id,
                    "metadata": metadata,
                    "code": code
                }))
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
    top_nodes = unique_results[:len(embeddings_input)]
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
        "top": 1
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
                ids=list(all_neighbors_ids),
                include=["documents", "metadatas"]
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
                score = meta.get("importance", {}).get("combined", 0.0) if isinstance(meta.get("importance"), dict) else meta.get("combined", 0.0)
                if doc:
                    importance_scores.append((score, nid, doc))
            sorted_by_importance = sorted(importance_scores, key=lambda x: -x[0])
            fallback_docs = [doc for _, _, doc in sorted_by_importance[:5]]
            full_context = "\n\n".join(fallback_docs)
        except Exception as e:
            logger.error(f"Error retrieving fallback for general question: {e}")
            full_context = "<NO CONTEXT FOUND>"
    return top_nodes, full_context or "<NO CONTEXT FOUND>"
