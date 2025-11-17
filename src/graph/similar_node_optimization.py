import json
import re
import time
from functools import lru_cache
from typing import List, Tuple, Dict, Any, Optional, Set

from src.clients.chroma_client import get_collection
from src.core.intent_analyzer import get_intent_analyzer, IntentCategory
from src.clients.llm_client import call_llm
from loguru import logger

max_cache_size = 1000
max_usage_results = 10
max_top_nodes_for_usage = 7
max_usage_nodes_for_context = 5
max_public_interfaces = 2
max_other_usage = 1
max_code_preview = 50
debug_results_limit = 5
max_nodes_limit = 1000
code_hash_length = 200
max_test_usage = 4
max_describe_nodes = 1

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
        logger.info("Ładuje model grafu")
        from src.graph.generate_embeddings_graph import generate_embeddings_graph
        logger.info("Model grafu jest gotowy")
    return True


@lru_cache(maxsize=max_cache_size)
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


def rerank_results(query: str, nodes: List[Tuple[float, Dict[str, Any]]], analyses: Dict[str, Any]) -> List[Tuple[float, Dict[str, Any]]]:
    """
        Reranks retrieved nodes.

        Adjusts base similarity scores based on detected intent/category (e.g., testing,
        usage, definition, implementation, exception), textual overlap with the query,
        structural importance metrics (PageRank, in-degree, combined), code length,
        and simple domain cues (e.g., controller/service/repository).

        Args:
            query (str): Original user query text.
            nodes (List[Tuple[float, Dict[str, Any]]]): Retrieved items as (score, node_data),
                where node_data includes 'node', 'metadata', and 'code'.
            analyses (Dict[str, Any]): Analysis payload containing at least:
                - category (str): Detected category name.
                - confidence (float): Confidence in [0.0, 1.0].

        Returns:
            List[Tuple[float, Dict[str, Any]]]: Reranked items sorted by adjusted score (desc).

        Notes:
            - Falls back to GENERAL intent if category is invalid.
    """
    category = analyses.get("category", "general")
    confidence = analyses.get("confidence", 0.5)
    try:
        intent_category = IntentCategory(category)
    except ValueError:
        intent_category = IntentCategory.GENERAL
    target_class_name = None
    if intent_category == IntentCategory.TESTING:
        import re
        class_match = re.search(r'for\s+(\w+)\s+class', query.lower())
        if class_match:
            target_class_name = class_match.group(1).lower()
        else:
            class_patterns = [
                r'(\w+controller)\s+class',
                r'(\w+service)\s+class',
                r'(\w+repository)\s+class',
                r'tests?\s+for\s+(\w+)',
                r'(\w+)\s+tests?',
                r'test.*(\w+controller)',
                r'test.*(\w+service)'
            ]
            for pattern in class_patterns:
                match = re.search(pattern, query.lower())
                if match:
                    target_class_name = match.group(1).lower()
                    break

    if target_class_name:
        logger.debug(f"Testing category, target class: {target_class_name}")
    reranked = []
    for item in nodes:
        if isinstance(item, tuple) and len(item) == 2:
            score, node_data = item
        else:
            score = item.get("score", 0.5)
            node_data = item

        node_id = node_data.get("node", "")
        metadata = node_data.get("metadata", {})
        code = node_data.get("code", "")
        kind = metadata.get("kind", "")
        label = metadata.get("label", "")

        adjusted_score = float(score)

        if intent_category == IntentCategory.TESTING:
            if target_class_name and target_class_name in node_id.lower():
                if kind == "METHOD" and "test" in node_id.lower():
                    adjusted_score *= 10.0
                    logger.debug(f"boost for test method: {node_id} - {adjusted_score:.3f}")
                elif kind == "CLASS" and "test" in node_id.lower():
                    adjusted_score *= 8.0
                    logger.debug(f"boost for test class: {node_id} - {adjusted_score:.3f}")
                elif kind == "METHOD" and any(test_word in label.lower() for test_word in ["should", "test"]):
                    adjusted_score *= 7.0
                    logger.debug(f"test method boost: {node_id} - {adjusted_score:.3f}")

            if "test" in node_id.lower():
                if kind == "METHOD":
                    adjusted_score *= 3.0
                elif kind == "CLASS":
                    adjusted_score *= 2.5
            elif "@test" in code.lower():
                adjusted_score *= 2.0
            elif kind == "METHOD" and ("should" in label.lower() or "test" in label.lower()):
                adjusted_score *= 1.8

            if ("test" not in node_id.lower() and
                    "@test" not in code.lower() and
                    not any(test_word in label.lower() for test_word in ["should", "test"])):
                adjusted_score *= 0.2

        elif intent_category == IntentCategory.USAGE:
            if "controller" in node_id.lower() and "test" not in node_id.lower():
                adjusted_score *= 2.0
            elif "service" in node_id.lower():
                adjusted_score *= 1.5
            elif kind == "METHOD" and any(annotation in code for annotation in
                                          ["@GetMapping", "@PostMapping", "@PutMapping", "@DeleteMapping"]):
                adjusted_score *= 1.8
            elif "repository" in node_id.lower():
                adjusted_score *= 1.3

        elif intent_category == IntentCategory.DEFINITION:
            if kind == "CLASS":
                adjusted_score *= 1.8
            elif kind == "INTERFACE":
                adjusted_score *= 1.6
            elif kind == "CONSTRUCTOR":
                adjusted_score *= 1.4
            elif kind == "METHOD" and label.lower() in ["main", "init", "setup"]:
                adjusted_score *= 1.3
            elif kind == "VARIABLE" and "final" in code.lower():
                adjusted_score *= 0.9

        elif intent_category == IntentCategory.IMPLEMENTATION:
            if kind == "METHOD":
                adjusted_score *= 1.5
            elif kind == "CONSTRUCTOR":
                adjusted_score *= 1.3
            elif kind == "CLASS" and "abstract" in code.lower():
                adjusted_score *= 1.2
            elif kind == "VARIABLE":
                adjusted_score *= 0.8

        elif intent_category == IntentCategory.EXCEPTION:
            if "exception" in node_id.lower() or "error" in node_id.lower():
                adjusted_score *= 2.0
            elif kind == "CLASS" and ("Exception" in label or "Error" in label):
                adjusted_score *= 1.8
            elif "throw" in code.lower() or "catch" in code.lower():
                adjusted_score *= 1.5

        if len(code) < 100:
            adjusted_score *= 0.7
        elif len(code) > 1000:
            adjusted_score *= 1.1

        importance = metadata.get("importance", {})
        if isinstance(importance, dict):
            combined_score = importance.get("combined", 0.0)
            if combined_score > 10.0:
                adjusted_score *= 1.4
            elif combined_score > 5.0:
                adjusted_score *= 1.2
            elif combined_score > 2.0:
                adjusted_score *= 1.1

            pagerank = importance.get("pagerank", 0.0)
            if pagerank > 0.01:
                adjusted_score *= 1.2
            elif pagerank > 0.005:
                adjusted_score *= 1.1

            in_degree = importance.get("in-degree", 0.0)
            if in_degree > 5:
                adjusted_score *= 1.3
            elif in_degree > 2:
                adjusted_score *= 1.1

        query_terms = set(query.lower().split())
        label_terms = set(label.lower().split())
        node_id_terms = set(node_id.lower().replace(".", " ").replace("(", " ").replace(")", " ").split())

        term_overlap = len(query_terms.intersection(label_terms.union(node_id_terms)))
        if term_overlap > 0:
            adjusted_score *= (1.0 + 0.15 * term_overlap)

        if kind == "PARAMETER" or kind == "VARIABLE":
            adjusted_score *= 0.6

        if category != "testing" and "test" in node_id.lower():
            adjusted_score *= 0.8

        if confidence > 0.7:
            adjusted_score *= (1.0 + (confidence - 0.7) * 0.5)

        reranked.append((adjusted_score, node_data))

    reranked.sort(key=lambda x: x[0], reverse=True)

    if intent_category == IntentCategory.TESTING:
        logger.debug(f"\nTop 10 reranked results for testing:")
        for i, (score, node_data) in enumerate(reranked[:10]):
            node_id = node_data.get("node", "")
            kind = node_data.get("metadata", {}).get("kind", "")
            logger.debug(f"{i + 1}. {node_id} ({kind}) - Score: {score:.3f}")

    return reranked


def find_usage_nodes(collection: Any, target_class_name: str, max_results: int = max_usage_results) -> List[Tuple[float, str, str, Dict[str, Any]]]:
    """
        Finds code nodes that use a given class/service (e.g., calls, endpoints, tests).

        Scans a Chroma collection for documents mentioning `target_class_name` and detect
        method/service calls, deduplicate similar snippets, and keep a
        balanced mix (controllers, services, tests, others).

        Args:
            collection (Any): Chroma collection handle with `get()` & `query()`-like API.
            target_class_name (str): Class or service name to search usages for.
            max_results (int, optional): Maximum number of usage examples to return.
                Defaults to `max_usage_results`.

        Returns:
            List[Tuple[float, str, str, Dict[str, Any]]]: Sorted usage items as tuples:
                (score, node_id, code, metadata).

        Notes:
            - Skips low-signal kinds (PARAMETER/VARIABLE/VALUE/IMPORT).
            - Boosts HTTP endpoints (@GetMapping/@PostMapping/...) for controller usage.
            - Performs simple de-duplication using a hash of the snippet prefix.
    """
    logger.debug(f"Szukam węzłów, które używają: {target_class_name}")
    try:
        all_nodes = collection.get(
            limit=max_nodes_limit,
            include=["metadatas", "documents"]
        )

        usage_nodes = []
        seen_patterns = set()
        target_patterns = [target_class_name.lower(), target_class_name]

        for i in range(len(all_nodes["ids"])):
            node_id = all_nodes["ids"][i]
            metadata = all_nodes["metadatas"][i]
            doc = all_nodes["documents"][i]

            if not doc or doc.startswith('<'):
                continue
            if target_class_name.lower() in node_id.lower() and metadata.get('kind') == 'METHOD':
                is_test_method = 'test' in node_id.lower()
                is_controller_endpoint = 'controller' in node_id.lower() and any(
                    annotation in doc for annotation in ['@GetMapping', '@PostMapping', '@PutMapping', '@DeleteMapping', '@RequestMapping'])

                if not is_test_method and not is_controller_endpoint:
                    logger.debug(f"Pominięto definicję metody: {node_id}")
                    continue
                else:
                    logger.debug(f"Keeping as usage example: {node_id}")

            kind = metadata.get('kind', '')
            if kind in ['PARAMETER', 'VARIABLE', 'VALUE', 'IMPORT']:
                continue

            found_usage = False
            score = 0.5
            usage_type = None
            pattern_key = None

            if f'{target_class_name}(' in doc:
                call_patterns = [
                    f'= {target_class_name}(',
                    f' {target_class_name}(',
                    f'return {target_class_name}(',
                    f'.{target_class_name}(',
                    f'({target_class_name}(',
                ]

                is_method_call = any(pattern in doc for pattern in call_patterns)
                method_definition_patterns = [
                    f'public {target_class_name}(',
                    f'private {target_class_name}(',
                    f'protected {target_class_name}(',
                ]
                is_definition = any(pattern in doc for pattern in method_definition_patterns)

                if is_method_call and not is_definition:
                    found_usage = True
                    score += 0.4
                    usage_type = "method_call"
                    pattern_key = f"method_call_{target_class_name}"
                    logger.debug(f"Method call: {target_class_name}() in {node_id}")

            elif f'.{target_class_name}(' in doc.lower():
                found_usage = True
                score += 0.5
                usage_type = "service_call"
                pattern_key = f"service_call_{target_class_name}"
                logger.debug(f"Service call: .{target_class_name}() in {node_id}")

            if found_usage and pattern_key:
                code_hash = hash(doc[:code_hash_length])
                unique_key = f"{pattern_key}_{code_hash}"

                if unique_key in seen_patterns:
                    logger.debug(f"Pominięto duplikat: {node_id}")
                    continue

                seen_patterns.add(unique_key)

            if found_usage and usage_type in ['method_call', 'service_call']:
                if 'controller' in node_id.lower() and 'test' not in node_id.lower():
                    if any(mapping in doc for mapping in
                           ['@GetMapping', '@PostMapping', '@PutMapping', '@DeleteMapping']):
                        score += 0.3
                        logger.debug(f"HTTP controller endpoint in {node_id}")

            if found_usage:
                if kind == 'CONSTRUCTOR' and usage_type != "method_call":
                    logger.debug(f"Pominięto constructor bez method call: {node_id}")
                    continue

                usage_nodes.append((score, node_id, doc, metadata))

        usage_nodes.sort(key=lambda x: x[0], reverse=True)

        filtered_usage = []
        controller_count = 0
        service_count = 0
        other_count = 0
        test_count = 0

        for score, node_id, doc, metadata in usage_nodes:
            node_lower = node_id.lower()

            if 'test' in node_lower and test_count < max_test_usage:
                kind = metadata.get('kind', '')
                if kind == 'CLASS':
                    test_count += 1
                filtered_usage.append((score, node_id, doc, metadata))
            elif 'controller' in node_lower and controller_count < max_public_interfaces:
                filtered_usage.append((score, node_id, doc, metadata))
                controller_count += 1
            elif 'service' in node_lower and service_count < max_public_interfaces:
                filtered_usage.append((score, node_id, doc, metadata))
                service_count += 1
            elif other_count < max_other_usage:
                filtered_usage.append((score, node_id, doc, metadata))
                other_count += 1

            if len(filtered_usage) >= max_results:
                break

        logger.debug(f"Znaleziono {len(filtered_usage)} użyć po filtrowaniu (z {len(usage_nodes)} pierwotnych)")

        if filtered_usage:
            logger.debug("Top usage nodes (po filtrowaniu):")
            for i, (score, node_id, doc, metadata) in enumerate(filtered_usage[:debug_results_limit]):
                logger.debug(f"{i + 1}. {node_id} (score: {score:.3f})")
                logger.debug(f"Kind: {metadata.get('kind', 'UNKNOWN')}")
                logger.debug(f"Preview: {doc[:100]}...")

        return filtered_usage

    except Exception as e:
        logger.error(f"Error in find_usage_nodes: {e}")
        return []


async def general_question(question, collection, top_k=5, max_neighbors=3, code_snippet_limit=800, batch_size=5):
    """
        Retrieves top nodes for a general question using LLM-guided coarse filtering.

        1) Asks an LLM to propose relevant node KINDS and name KEYWORDS.
        2) Filters candidates by kind/keywords and blends with importance (`combined`).
        3) Batches candidate snippets and asks the LLM to score relevance (1–5).
        4) Expands with related neighbors and usage examples of selected classes.
        5) Returns the top-k nodes with metadata and code.

        Args:
            question (str): Natural-language user question.
            collection: Chroma collection handle.
            top_k (int, optional): Number of final nodes to return. Defaults to 5.
            max_neighbors (int, optional): Max related neighbors to fetch per selected node.
                Defaults to 3.
            code_snippet_limit (int, optional): Max characters per snippet sent to LLM.
                Defaults to 500.
            batch_size (int, optional): Number of candidates scored per LLM batch.
                Defaults to 5.

        Returns:
            List[Tuple[int, Dict[str, Any]]]: Top nodes as (score, node_data) where
            node_data contains `node`, `metadata`, and `code`.
        """
    kind_weights = {
        "CLASS": 2.0,
        "INTERFACE": 1.8,
        "METHOD": 1.0,
        "CONSTRUCTOR": 1.2,
        "VARIABLE": 0.8,
        "PARAMETER": 0.5,
    }

    classification_prompt = f"""
    Pytanie użytkownika: "{question}"
    Twoje zadanie:
    1. Określ, jakie typy węzłów (CLASS, METHOD, VARIABLE, PARAMETER, CONSTRUCTOR) mogą być najbardziej istotne.
    2. Podaj 5-10 słów kluczowych, które powinny występować w nazwach tych węzłów.
    Odpowiedz tylko w postaci JSON, np.:
    {{"kinds": ["CLASS", "METHOD"], "keywords": ["frontend","controller","view"]}}
    Żadnych komentarzy
    """
    analysis = await call_llm(classification_prompt)
    logger.debug(f"LLM analysis: {analysis}")
    try:
        analysis = json.loads(analysis)
    except:
        analysis = {"kinds": [], "keywords": []}

    kinds = set([k.upper() for k in analysis.get("kinds", [])])
    keywords = [kw.lower() for kw in analysis.get("keywords", [])]

    all_nodes = collection.get(include=["metadatas", "documents"])

    logger.debug(f"kinds: {kinds}, keywords: {keywords}")

    candidate_nodes = []
    for i in range(len(all_nodes["ids"])):
        node_id = all_nodes["ids"][i]
        metadata = all_nodes["metadatas"][i]
        doc = all_nodes["documents"][i] or ""
        kind = metadata.get("kind", "").upper()

        if kinds and kind not in kinds:
            continue

        score = 1 if (not kinds or kind in kinds) else 0
        for kw in keywords:
            if kw in node_id.lower():
                score += kind_weights.get(kind, 1.0)

        if score == 0:
            score = 0.1

        combined_score = float(metadata.get("combined", 0.0))
        hybrid_score = score * 1000 + combined_score
        candidate_nodes.append((node_id, metadata, doc, hybrid_score))

    if not candidate_nodes:
        logger.debug("Brak kandydatów, wybieram fallback top-5 wg combined")
        fallback_nodes = sorted(
            zip(all_nodes["ids"], all_nodes["metadatas"], all_nodes["documents"]),
            key=lambda x: float(x[1].get("combined", 0.0)),
            reverse=True
        )[:top_k]
        return [(1, {"node": nid, "metadata": meta, "code": doc}) for nid, meta, doc in fallback_nodes]

    candidates_sorted = sorted(candidate_nodes, key=lambda x: x[3], reverse=True)[:top_k * 2]

    top_nodes = []
    seen_nodes = set()
    for i in range(0, len(candidates_sorted), batch_size):
        batch = candidates_sorted[i:i + batch_size]
        code_snippets_map = []
        for node_id, meta, doc, hybrid_score in batch:
            snippet = "\n".join(doc.splitlines())[:code_snippet_limit]
            code_snippets_map.append({"id": node_id, "code": snippet})

        prompt = f"""
        Pytanie: '{question}'

        Oceń każdy fragment kodu od 1 do 5:
        1 = nie pasuje wcale
        3 = fragment średnio pasuje, ale cały kod sądzać po fragmencie powinien pomóc
        5 = dokładnie odpowiada

        Zwróć JSON: [{{"id": "node_id", "score": 3}}, ...]
        Żadnych komentarzy, ani wyjaśnień.

        Fragmenty:
        {json.dumps(code_snippets_map, indent=2)}
        """
        answer = await call_llm(prompt)
        print(answer)
        clean_answer = re.sub(r"```(?:json)?", "", answer).strip()
        try:
            scores = json.loads(clean_answer)
        except:
            scores = []
        logger.debug(f"LLM scores: {scores}")
        for s in scores:
            node_id = s.get("id")
            score = int(s.get("score", 0))
            if score >= 3:
                node_tuple = next((c for c in batch if c[0] == node_id), None)
                if node_tuple:
                    _, metadata, doc, _ = node_tuple
                    if node_id not in seen_nodes:
                        top_nodes.append((score, {"node": node_id, "metadata": metadata, "code": doc}))
                        seen_nodes.add(node_id)

                    related_entities_str = metadata.get("related_entities", "")
                    try:
                        related_entities = json.loads(related_entities_str) if isinstance(related_entities_str,
                                                                                          str) else related_entities_str
                    except:
                        related_entities = []

                    neighbors_to_fetch = [nid for nid in related_entities[:max_neighbors] if nid not in seen_nodes]

                    if not neighbors_to_fetch:
                        continue

                    neighbors = collection.get(ids=neighbors_to_fetch, include=["metadatas", "documents"])
                    for j in range(len(neighbors["ids"])):
                        neighbor_id = neighbors["ids"][j]
                        if neighbor_id in seen_nodes:
                            continue
                        neighbor_metadata = neighbors["metadatas"][j]
                        neighbor_kind = neighbor_metadata.get("kind", "")
                        neighbor_doc = neighbors["documents"][j] or ""

                        # pomijamy sasiadow metody i zmienne, ktorych kod juz nalezy do klasy rodzica
                        if metadata.get("kind") == "CLASS" and (
                                neighbor_kind == "METHOD" or neighbor_kind == "VARIABLE") and str(
                                neighbor_id).startswith(f"{node_id}."):
                            continue

                        top_nodes.append((score - 1, {
                            "node": neighbor_id,
                            "metadata": neighbor_metadata,
                            "code": neighbor_doc
                        }))
                        seen_nodes.add(neighbor_id)


    final_top_nodes = top_nodes.copy()
    for score, node_data in top_nodes:
        if node_data["metadata"].get("kind") == "CLASS":
            class_name = node_data["metadata"].get("label")
            usage_nodes = find_usage_nodes(collection, class_name, max_results=max_usage_nodes_for_context)
            for u_score, u_node_id, u_doc, u_metadata in usage_nodes:
                if u_node_id in seen_nodes:
                    continue
                final_top_nodes.append((u_score - 1, {
                    "node": u_node_id,
                    "metadata": u_metadata,
                    "code": u_doc
                }))
                seen_nodes.add(u_node_id)

    top_nodes = sorted(final_top_nodes, key=lambda x: x[0], reverse=True)[:top_k]
    logger.debug(f"TOP NODES from general_question: {[n[1]['node'] for n in top_nodes]}")

    return top_nodes


def get_metric_value(node, metric):
    """
        Returns a numeric metric value for a node metadata record.

        Supports a special derived metric `number_of_neighbors` by counting
        `related_entities`. Falls back to a float cast of the requested metric.

        Args:
            node (dict): Node metadata dictionary.
            metric (str): Metric key (e.g., "combined", "pagerank",
                "in-degree", "out-degree", "number_of_neighbors").

        Returns:
            float: Metric value for the node.
    """
    if metric == "number_of_neighbors":
        related_entities_str = node.get("related_entities", "")
        try:
            related_entities = json.loads(related_entities_str) if isinstance(related_entities_str,
                                                                              str) else related_entities_str
        except:
            related_entities = []
        return len(related_entities)
    else:
        return float(node.get(metric, 0.0))


async def find_top_nodes(question, collection):
    """
        Finds top nodes based on LLM-guided kind/metric selection.

        Asks an LLM to determine relevant node kinds, a ranking metric, a result
        limit, and sort order, then filters and sorts nodes accordingly.

        Args:
            question (str): Natural-language question guiding the selection.
            collection: Chroma collection handle used to fetch node metadata.

        Returns:
            List[dict]: Top nodes as dictionaries with keys:
                - node (str): Node ID.
                - metadata (dict): Node metadata.
                - metric_value (float): Value used for sorting.
    """
    classification_prompt = f"""
        Pytanie użytkownika: "{question}"
        Twoje zadanie:
        1. Określ, jakich typów węzłów (CLASS, METHOD, VARIABLE, PARAMETER, CONSTRUCTOR) dotyczy pytanie.
        2. Dopasuj klasyfikator do pytania spośród: loc (lines of code - rozmiar kodu), pagerank, eigenvector, in_degree, out_degree, combined, number_of_neighbors.
        3. Określ ile węzłów użytkownik chce dostać.
        4. Zdecyduj czy wybrać największe wartości (malejąco - "desc") czy najmniejsze wartości (rosnąco - "asc"). 
               - Jeśli pytanie zawiera słowa typu "największe", "biggest", "largest", "most", "max" to ustaw "desc".
               - Jeśli pytanie zawiera słowa typu "najmniejsze", "najmniej", "least", "smallest", "min" to ustaw "asc".
        5. Jeśli pytanie dotyczy ogólnie "najważniejszych", "kluczowych", "głównych" lub "centralnych" elementów,
            wybierz "CLASS" oraz "combined".
        
        
        Odpowiedz wyłącznie w postaci JSON, np.:
        {{"kinds": ["CLASS", "METHOD"], "metric": "combined", "limit": 5, "order": "asc"}}
        Żadnych komentarzy, tylko JSON.
        """

    analysis = await call_llm(classification_prompt)
    logger.debug(analysis)

    try:
        parsed = json.loads(analysis)
        kinds = parsed.get("kinds", ["CLASS"])
        metric = parsed.get("metric", "combined")
        order = parsed.get("order", "desc")
        limit = parsed.get("limit", 5)
    except json.JSONDecodeError:
        kinds = ["CLASS"]
        metric = "combined"
        order = "desc"
        limit = 5

    results = collection.get(include=["metadatas"])

    nodes = [
        {
            "node": results["ids"][i],
            "metadata": results["metadatas"][i],
            "metric_value": get_metric_value(results["metadatas"][i], metric),
        }
        for i in range(len(results["ids"]))
    ]

    logger.debug(nodes[0])

    filtered_sorted_nodes = sorted(
        (node for node in nodes if node["metadata"].get("kind") in kinds),
        key=lambda n: n["metric_value"],
        reverse=(order.lower() == "desc")
    )

    return filtered_sorted_nodes[:limit]


async def similar_node_fast(question: str, model_name: str = "microsoft/codebert-base", top_k: int = 20) -> Tuple[
    List[Tuple[float, Dict[str, Any]]], str]:
    """
        Fast path for retrieving similar nodes and building a final context.

        Mixes intent analysis, embedding search (CodeBERT + Chroma), LLM-guided
        filtering for general queries, and heuristic re-ranking. Optionally returns
        a “top” summary using metric-based selection. Produces a token-bounded,
        intent-aware context string.

        Args:
            question (str): User’s natural-language question.
            model_name (str, optional): Embedding model identifier. Defaults to CodeBERT.
            top_k (int, optional): Base number of results per query embedding. Defaults to 20.

        Returns:
            Tuple[List[Tuple[float, Dict[str, Any]]], str]:
                - List of (score, node_data) tuples (deduplicated, reranked).
                - Built context string (or a fallback marker).

        Notes:
            - Uses exception-specific embedding search when category == "exception".
            - For category "top" with sufficient confidence, selects by metric/kind via `find_top_nodes`.
            - Applies `rerank_results` and `build_context` for final assembly.
        """
    start_time = time.time()
    try:
        from src.graph.retriver import chroma_client, extract_key_value_pairs_simple
        from src.graph.generate_embeddings_graph import generate_embeddings_graph

        try:
            from context import build_context
        except ImportError:
            def build_context(nodes, category, confidence, question="", target_method=None):
                return "\n\n".join([node[1]["code"] for node in nodes[:5] if node[1]["code"]])

        collection = get_collection("scg_embeddings")
        pairs = await extract_key_value_pairs_simple(question)
        logger.debug(f"Pytanie: '{question}'")
        logger.debug(f"Wyciągnięte pary: {pairs}")

        embeddings_input = []
        for key, value in pairs:
            embeddings_input.append(f"{key} {value}" if key else value)

        if not embeddings_input:
            embeddings_input = [question]
        logger.debug(f"Embedding input: {embeddings_input}")

        get_graph_model()
        analyzer = get_intent_analyzer()
        analysis_result = analyzer.enhanced_classify_question(question)
        analysis = {
            "category": analysis_result.primary_intent.value,
            "confidence": analysis_result.confidence,
            "scores": analysis_result.scores,
            "enhanced": analysis_result.enhanced
        }

        if analysis["category"] == "top" and analysis["confidence"] > 0.6:
            logger.debug("Finding top classes or methods")
            top_nodes = await find_top_nodes(question, collection)
            context = ''
            context = " ".join(
                f"{node.get('metadata', {}).get('label', '')} - {node.get('metric_value'):.2f}" for node in top_nodes
            )
            logger.debug(context)
            end_time = time.time()
            elapsed_ms = (end_time - start_time) * 1000
            logger.debug(f"ukonczono w: {elapsed_ms:.1f}ms")
            return top_nodes, context or "<NO CONTEXT FOUND>"


        logger.debug(f"Enhanced classification: {analysis}")
        if not pairs or (analysis["category"] == "general" and analysis["confidence"] > 0.6):
            if analysis["category"] == "exception":
                logger.debug("EXCEPTION category detected - forcing embeddings search")
                embeddings_input = [question]
                query_embeddings = generate_embeddings_graph(embeddings_input, model_name)
                query_result = collection.query(
                    query_embeddings=[query_embeddings[0].tolist()],
                    n_results=top_k * 2,
                    include=["embeddings", "metadatas", "documents", "distances"]
                )

                all_results = []
                for i in range(len(query_result["ids"][0])):
                    score = 1 - query_result["distances"][0][i]
                    node_id = query_result["ids"][0][i]
                    metadata = query_result["metadatas"][0][i]
                    code = query_result["documents"][0][i]
                    all_results.append((score, {
                        "node": node_id,
                        "metadata": metadata,
                        "code": code}))
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
                top_nodes,
                category,
                confidence,
                question=question,
                target_method=None
            )
            end_time = time.time()
            elapsed_ms = (end_time - start_time) * 1000
            logger.debug(f"ukonczono w: {elapsed_ms:.1f}ms")
            return top_nodes, full_context or "<NO CONTEXT FOUND>"

        query_embeddings = generate_embeddings_graph(embeddings_input, model_name)
        all_results = []
        batch_embeddings = [emb.tolist() for emb in query_embeddings]

        for i, emb in enumerate(batch_embeddings):
            if len(batch_embeddings) > 1:
                logger.debug(f"  Zapytanie {i + 1}/{len(batch_embeddings)}: '{embeddings_input[i]}'")
            else:
                logger.debug("Proste zapytanie 1 embedding")
            query_result = collection.query(
                query_embeddings=[emb],
                n_results=1,
                include=["embeddings", "metadatas", "documents", "distances"]
            )

            for j in range(len(query_result["ids"][0])):
                score = 1 - query_result["distances"][0][j]
                node_id = query_result["ids"][0][j]
                metadata = query_result["metadatas"][0][j]
                code = query_result["documents"][0][j]

                if j < debug_results_limit:
                    raw_distance = query_result["distances"][0][j]
                    calculated_score = 1 - raw_distance
                    logger.debug(f"Wynik {j + 1}:")
                    logger.debug(f"Node ID: {node_id}")
                    logger.debug(f"Raw distance: {raw_distance:.4f}")
                    logger.debug(f"Calculated score: {calculated_score:.4f}")
                    logger.debug(f"Label: {metadata.get('label', 'NO_LABEL')}")
                    logger.debug(f"Kind: {metadata.get('kind', 'NO_KIND')}")
                    logger.debug("")

                    all_results.append((score, {
                        "node": node_id,
                        "metadata": metadata,
                        "code": code
                    }))

        logger.debug(f"Zebrano łącznie {len(all_results)} wyników z ChromaDB")

        reranked_results = rerank_results(question, all_results, analysis)
        logger.debug(f"Reranked {len(reranked_results)} results")

        logger.debug(f"Deduplikowanie: usuwam duplikaty z {len(reranked_results)} wyników")
        seen: Set[str] = set()
        unique_results = []
        for score, node in reranked_results:
            node_id = node["node"]
            if node_id not in seen:
                unique_results.append((score, node))
                seen.add(node_id)
                if len(unique_results) >= len(embeddings_input) * top_k:
                    break
        logger.debug(f"Po deduplikowaniu: {len(unique_results)} unikalnych wyników")

        if analyzer.is_usage_question(question):
            logger.debug("Usage question. Szukam w related_entities")

            target_entity = None
            target_type = None
            if all_results:
                best_match = all_results[0]
                node_id = best_match[1]["node"]
                metadata = best_match[1]["metadata"]
                logger.debug(f"Using ORIGINAL best match from ChromaDB: {node_id}")

                if metadata.get("kind") == "METHOD":
                    target_entity = metadata.get("label")
                    target_type = "method"
                    logger.debug(f"Target method identified: {target_entity}")
                elif metadata.get("kind") == "CLASS":
                    target_entity = metadata.get("label")
                    target_type = "class"
                    logger.debug(f"Target class identified: {target_entity}")
                elif "." in node_id:
                    parts = node_id.split('.')
                    for part in reversed(parts):
                        if part and part[0].isupper():
                            target_entity = part
                            target_type = "class"
                            logger.debug(f"Target class identified from node_id: {target_entity}")
                            break

            top_nodes = unique_results[:1]

            if target_entity:
                usage_nodes = find_usage_nodes(collection, target_entity, max_results=max_usage_nodes_for_context)

                for score, node_id, doc, metadata in usage_nodes:
                    top_nodes.append((score, {
                        "node": node_id,
                        "metadata": metadata,
                        "code": doc
                    }))

                logger.debug(f"Added {len(usage_nodes)} usage nodes to results")

            top_nodes = top_nodes[:max_top_nodes_for_usage]

        else:
            # Żeby było mniej kontekstu dla definicji bierzemy tyle max_describe_nodes na podstawie zlozonosci pytania
            # i dobieramy 2 sasiadow dla kazdego
            top_nodes = unique_results[:len(embeddings_input)]
            """
                We take neighbors only if question is about definiton of one object, otherwise too much context
                and llm may have problems with answering.
                If user wants more info about one class he should ask about that one class
            """
            if len(top_nodes) == 1:
                current_nodes = top_nodes.copy()
                added_nodes = {node_data["node"] for _, node_data in top_nodes}
                for node_score, node_data in current_nodes:
                    node_id = node_data["node"]
                    metadata = node_data["metadata"]

                    related_entities_str = metadata.get("related_entities", "")
                    try:
                        related_entities = json.loads(related_entities_str) if isinstance(related_entities_str,
                                                                                          str) else related_entities_str
                    except:
                        related_entities = []

                    neighbors_to_fetch = related_entities[:2]

                    if neighbors_to_fetch:
                        try:
                            neighbors = collection.get(ids=neighbors_to_fetch, include=["metadatas", "documents"])
                        except Exception as e:
                            logger.warning(f"ChromaDB get failed (likely telemetry): {e}")
                            neighbors = {"ids": [], "metadatas": [], "documents": []}
                        for j in range(len(neighbors["ids"])):
                            neighbor_id = neighbors["ids"][j]
                            neighbor_metadata = neighbors["metadatas"][j]
                            neighbor_kind = neighbor_metadata.get("kind", "")
                            neighbor_doc = neighbors["documents"][j] or ""

                            # pomijamy sasiadow metody i zmienne, ktorych kod juz nalezy do klasy rodzica
                            if metadata.get("kind") == "CLASS" and (
                                    neighbor_kind == "METHOD" or neighbor_kind == "VARIABLE") and str(
                                neighbor_id).startswith(f"{node_id}.") or neighbor_id in added_nodes:
                                continue

                            top_nodes.append((node_score, {
                                "node": neighbor_id,
                                "metadata": neighbor_metadata,
                                "code": neighbor_doc
                            }))
                            added_nodes.add(neighbor_id)

        logger.debug(f"Wybrano {len(top_nodes)} najlepszych węzłów")

        category = analysis.get("category", "general")
        confidence = analysis.get("confidence", 0.5)
        logger.debug(f"Building context with category={category}, confidence={confidence}, target={target_entity if 'target_entity' in locals() else None}")
        full_context = build_context(
            top_nodes,
            category,
            confidence,
            question=question,
            target_method=target_entity if 'target_entity' in locals() else None
        )

        logger.debug(f"Context built: {len(full_context)} chars")

        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        logger.debug(f"ukonczono w: {elapsed_ms:.1f}ms")
        return top_nodes, full_context or "<NO CONTEXT FOUND>"

    except Exception as e:
        logger.warning(f"Fallback do oryginalnej funkcji: {e}")
        from src.graph.retriver import similar_node
        return similar_node(question, model_name, top_k)


_general_fallback_cache: Optional[str] = None
_cache_timestamp: float = 0
