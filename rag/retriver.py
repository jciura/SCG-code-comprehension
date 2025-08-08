import json
import logging as logger
import re

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from graph.generate_embeddings_graph import generate_embeddings_graph
from rag.generate_embeddings import generate_embeddings

with open("embeddings/classifier_example_embeddings.json", "r", encoding="utf-8") as f:
    classifier_embeddings = json.load(f)

classifier_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(
    path="embeddings/chroma_storage",
    settings=Settings(allow_reset=False)
)

def extract_file_name_from_question(question):
    file_patterns = re.findall(r'\b[\w-]+\.[\w]+\b', question.lower())
    words = [word.lower() for word in question.split()]
    return file_patterns, words


def calcuate_file_name_similiarity(file_name, target_patterns, target_words):
    file_name = file_name.lower()

    file_base = file_name.rsplit('.', 1)[0]
    if file_name in target_patterns:
        return 2.0
    if file_base in target_words:
        return 1.5

    for pattern in target_patterns:
        if pattern in file_name:
            return 1.3
    return 1.0


def normalize(v):
    return v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v

def similar_questions(question, qa_data, model_name="microsoft/codebert-base"):
    question_embedding = generate_embeddings([question], model_name)[0]
    results = []
    for qa in qa_data:
        qa_embedding = np.array(qa["embedding"])
        score = cosine_similarity([question_embedding], [qa_embedding])[0][0]
        results.append((score, qa))
    results.sort(reverse=True, key=lambda x: x[0])
    return results[:3]


def similar_code(question, code_path, model_name="microsoft/codebert-base", min_similarity=0.5):
    target_patterns, target_words = extract_file_name_from_question(question)
    logger.info(f"Extracted file patterns: {target_patterns}")
    logger.info(f"Extracted words that might be files: {target_words}")

    with open(code_path, "r") as f:
        code_data = [json.loads(line) for line in f]

    question_embedding = generate_embeddings([question], model_name)[0]
    results = []

    for chunk in code_data:
        chunk_embedding = np.array(chunk["embedding"])
        base_score = cosine_similarity([question_embedding], [chunk_embedding])[0][0]

        file_name = chunk['metadata']['file'].lower()
        file_muptiplier = calcuate_file_name_similiarity(file_name, target_patterns, target_words)
        final_score = base_score * file_muptiplier
        if file_muptiplier > 1.0:
            logger.info(
                f"File {file_name} score boosted from {base_score} to {final_score} (multiplier: {file_muptiplier})")

        if final_score >= min_similarity:
            results.append((final_score, chunk))
            logger.info(f"Added chunk from {file_name} with final score {final_score}")
        else:
            logger.debug(f"Skipped chunk from {file_name} - score {final_score} below threshold {min_similarity}")

    results.sort(reverse=True, key=lambda x: x[0])
    return results[:6]


def extract_key_value_pairs_simple(question):
    key_terms = {"class", "method", "function", "variable", "property"}
    words = question.lower().split()
    pairs = []

    for i, word in enumerate(words):
        if word in key_terms:
            pairs.append((word, words[i - 1]))

    if not pairs:
        pairs = [(None, w) for w in words if w not in key_terms and len(w) > 2]

    return pairs


def filter_repeated_codes(old_context, codes):
    filtered_codes = []
    for code in codes:
        if code not in old_context:
            filtered_codes.append(code)
    return filtered_codes


def classify_question(question):
    question_emb = classifier_model.encode([question], convert_to_tensor=False)[0]

    best_score = -1
    best_label = "general"

    for label, examples in classifier_embeddings.items():
        for emb in examples:
            score = cosine_similarity([question_emb], [emb])[0][0]
            if score > best_score:
                best_score = score
                best_label = label

    return best_label


def preprocess_question(q: str) -> str:
    q = re.sub(r'\bmethod\s+\w+\b', 'method', q, flags=re.IGNORECASE)
    q = re.sub(r'\bfunction\s+\w+\b', 'function', q, flags=re.IGNORECASE)
    q = re.sub(r'\bclass\s+\w+\b', 'class', q, flags=re.IGNORECASE)
    q = re.sub(r'\bvariable\s+\w+\b', 'variable', q, flags=re.IGNORECASE)

    q = re.sub(r'\s+', ' ', q).strip()

    return q.lower()


def similar_node(question, model_name="microsoft/codebert-base", top_k=7):
    collection = chroma_client.get_collection(name="scg_embeddings")
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
                include=["embeddings", "metadatas", "documents", "distances"]
            )

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
            print(f"Error querying collection: {e}")

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

    category = classify_question(preprocess_question(question))
    max_neighbors = {"general": 5, "medium": 3, "specific": 1}.get(category, 2)

    print(category)

    all_neighbors_ids = set()
    for _, node in top_nodes:
        neighbors = node["metadata"].get("related_entities", [])
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
            print(f"Error getting neighbors: {e}")

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
                score = meta.get("importance", {}).get("combined", 0.0)
                if doc:
                    importance_scores.append((score, nid, doc))
            sorted_by_importance = sorted(importance_scores, key=lambda x: -x[0])
            fallback_docs = [doc for _, _, doc in sorted_by_importance[:5]]
            full_context = "\n\n".join(fallback_docs)
        except Exception as e:
            print(f"Error retrieving fallback for general question: {e}")
            full_context = "<NO CONTEXT FOUND>"

    return top_nodes, full_context or "<NO CONTEXT FOUND>"
