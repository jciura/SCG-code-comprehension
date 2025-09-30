import time
from functools import lru_cache
from typing import List, Tuple, Dict, Any, Optional, Set
from chroma_client import get_collection


max_cache_size = 1000
max_usage_results = 10
max_top_nodes_for_usage = 3 #8
max_usage_nodes_for_context = 2 #7
max_usage_nodes_to_process = 5
max_public_interfaces = 2
max_tests = 1
max_other_usage = 1
max_definition_nodes = 2
max_context_preview = 1000
max_test_preview = 400
max_usage_preview = 600
max_definition_preview = 800
max_code_preview = 50
debug_results_limit = 5
max_nodes_limit = 1000
code_hash_length = 200
max_fallback_docs = 5
fallback_nodes_limit = 100
cache_max_age_seconds = 300


max_describe_nodes = 5
MAX_NEIGHBORS = {"general": 2, "medium": 1, "specific": 0}

_graph_model = None


def get_graph_model() -> bool:
    global _graph_model
    if _graph_model is None:
        print("Ładuje model grafu")
        from graph.generate_embeddings_graph import generate_embeddings_graph
        print("Model grafu jest gotowy")
    return True


@lru_cache(maxsize=max_cache_size)
def cached_question_question(question_hash: str) -> str:
    from graph.retriver import classify_question, preprocess_question
    return classify_question(preprocess_question(question_hash))



def rerank_results(query: str, nodes: List[Tuple[float, Dict[str, Any]]], analyses: Dict[str, Any]) -> List[
    Tuple[float, Dict[str, Any]]]:
    category = analyses.get("category", "general")
    confidence = analyses.get("confidence", 0.5)
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

        if category == "usage":
            if "controller" in node_id.lower() and "test" not in node_id.lower():
                adjusted_score *= 2.0
            elif "service" in node_id.lower():
                adjusted_score *= 1.5
            elif kind == "METHOD" and any(annotation in code for annotation in
                                          ["@GetMapping", "@PostMapping", "@PutMapping", "@DeleteMapping"]):
                adjusted_score *= 1.8
            elif "repository" in node_id.lower():
                adjusted_score *= 1.3

        elif category == "definition":
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

        elif category == "implementation":
            if kind == "METHOD":
                adjusted_score *= 1.5
            elif kind == "CONSTRUCTOR":
                adjusted_score *= 1.3
            elif kind == "CLASS" and "abstract" in code.lower():
                adjusted_score *= 1.2
            elif kind == "VARIABLE":
                adjusted_score *= 0.8

        elif category == "testing":
            if "test" in node_id.lower():
                adjusted_score *= 2.0
            elif kind == "METHOD" and "test" in label.lower():
                adjusted_score *= 1.8
            elif "mock" in code.lower():
                adjusted_score *= 1.4
            elif kind == "CLASS" and not "test" in node_id.lower():
                adjusted_score *= 0.7

        elif category == "exception":
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
    return reranked




def find_usage_nodes(collection: Any, target_class_name: str, max_results: int = max_usage_results) -> List[
    Tuple[float, str, str, Dict[str, Any]]]:
    print(f"Szukam węzłów, które używają: {target_class_name}")
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
                print(f"Pominięto definicję metody: {node_id}")
                continue

            kind = metadata.get('kind', '')
            if kind in ['PARAMETER', 'VARIABLE', 'VALUE', 'IMPORT']:
                # print(f"Pominięto {kind}: {node_id}")
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
                    print(f"Method call: {target_class_name}() in {node_id}")

            elif f'.{target_class_name}(' in doc.lower():
                found_usage = True
                score += 0.5
                usage_type = "service_call"
                pattern_key = f"service_call_{target_class_name}"
                print(f"Service call: .{target_class_name}() in {node_id}")

            if found_usage and pattern_key:
                code_hash = hash(doc[:200])
                unique_key = f"{pattern_key}_{code_hash}"

                if unique_key in seen_patterns:
                    print(f"Pominięto duplikat wzorca: {node_id}")
                    continue

                seen_patterns.add(unique_key)

            if found_usage and usage_type in ['method_call', 'service_call']:
                if 'controller' in node_id.lower() and 'test' not in node_id.lower():
                    if any(mapping in doc for mapping in
                           ['@GetMapping', '@PostMapping', '@PutMapping', '@DeleteMapping']):
                        score += 0.3
                        print(f"HTTP controller endpoint in {node_id}")

            if found_usage:
                if kind == 'CONSTRUCTOR' and usage_type != "method_call":
                    print(f"Pominięto constructor bez method call: {node_id}")
                    continue

                usage_nodes.append((score, node_id, doc, metadata))
        usage_nodes.sort(key=lambda x: x[0], reverse=True)

        filtered_usage = []
        controller_count = 0
        service_count = 0
        other_count = 0

        for score, node_id, doc, metadata in usage_nodes:
            node_lower = node_id.lower()


            if 'controller' in node_lower and 'test' not in node_lower and controller_count < 2:
                filtered_usage.append((score, node_id, doc, metadata))
                controller_count += 1
            elif 'service' in node_lower and 'test' not in node_lower and service_count < 2:
                filtered_usage.append((score, node_id, doc, metadata))
                service_count += 1
            elif other_count < 1 and 'controller' not in node_lower and 'service' not in node_lower:
                filtered_usage.append((score, node_id, doc, metadata))
                other_count += 1

            if len(filtered_usage) >= max_results:
                break

        print(f"Znaleziono {len(filtered_usage)} użyć po filtrowaniu (z {len(usage_nodes)} pierwotnych)")

        if filtered_usage:
            print("Top usage nodes (po filtrowaniu):")
            for i, (score, node_id, doc, metadata) in enumerate(filtered_usage[:debug_results_limit]):
                print(f"{i + 1}. {node_id} (score: {score:.3f})")
                print(f"Kind: {metadata.get('kind', 'UNKNOWN')}")
                print(f"Preview: {doc[:100]}...")

        return filtered_usage

    except Exception as e:
        print(f"Error in find_usage_nodes_improved: {e}")
        return []



def similar_node_fast(question: str, model_name: str = "microsoft/codebert-base", top_k: int = 20) -> Tuple[
    List[Tuple[float, Dict[str, Any]]], str]:
    start_time = time.time()
    try:
        from graph.retriver import chroma_client, extract_key_value_pairs_simple, enhanced_classify_question
        from graph.generate_embeddings_graph import generate_embeddings_graph

        try:
            from context import build_context
        except ImportError:
            def build_context(nodes, category, confidence):
                return "\n\n".join([node[1]["code"] for node in nodes[:5] if node[1]["code"]])

        collection = get_collection("scg_embeddings")
        pairs = extract_key_value_pairs_simple(question)
        print(f"Pytanie: '{question}'")
        print(f"Wyciągnięte pary: {pairs}")
        embeddings_input = []

        for key, value in pairs:
            embeddings_input.append(f"{key} {value}" if key else value)

        if not embeddings_input:
            embeddings_input = [question]
        print(f"Embedding input: {embeddings_input}")

        get_graph_model()
        query_embeddings = generate_embeddings_graph(embeddings_input, model_name)
        all_results = []

        if len(query_embeddings) == 1:
            print("Proste pytanie 1 embedding = 1 zapytanie do chromaDB")
            query_result = collection.query(
                query_embeddings=[query_embeddings[0].tolist()],
                n_results=top_k * len(embeddings_input),
                include=["embeddings", "metadatas", "documents", "distances"]
            )

            for i in range(len(query_result["ids"][0])):
                score = 1 - query_result["distances"][0][i]
                node_id = query_result["ids"][0][i]
                metadata = query_result["metadatas"][0][i]
                code = query_result["documents"][0][i]
                if i < debug_results_limit:
                    raw_distance = query_result["distances"][0][i]
                    calculated_score = 1 - raw_distance
                    print(f"Wynik {i + 1}:")
                    print(f"Node ID: {node_id}")
                    print(f"Raw distance: {raw_distance:.4f}")
                    print(f"Calculated score: {calculated_score:.4f}")
                    print(f"Label: {metadata.get('label', 'NO_LABEL')}")
                    print(f"Kind: {metadata.get('kind', 'NO_KIND')}")
                    print(f"Code preview: {code[:max_code_preview] if code else 'NO_CODE'}...")
                    print(" ")
                all_results.append((score, {
                    "node": node_id,
                    "metadata": metadata,
                    "code": code
                }))
        else:
            print(f"Złożone pytanie: {len(query_embeddings)} embeddingow = {len(query_embeddings)} zapytan do ChromaDB")

            batch_embeddings = [emb.tolist() for emb in query_embeddings]

            for i, emb in enumerate(batch_embeddings):
                print(f"  Zapytanie {i + 1}/{len(batch_embeddings)}: '{embeddings_input[i]}'")
                query_result = collection.query(
                    query_embeddings=[emb],
                    n_results=top_k,
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
                        print(f"Wynik {j + 1}:")
                        print(f"Node ID: {node_id}")
                        print(f"Raw distance: {raw_distance:.4f}")
                        print(f"Calculated score: {calculated_score:.4f}")
                        print(f"Label: {metadata.get('label', 'NO_LABEL')}")
                        print(f"Kind: {metadata.get('kind', 'NO_KIND')}")
                        print(" ")

                    all_results.append((score, {
                        "node": node_id,
                        "metadata": metadata,
                        "code": code
                    }))

        print(f"Zebrano lacznie {len(all_results)} wynikow z ChromaDb")

        analysis = enhanced_classify_question(question)
        print(f"Enhanced classification: {analysis}")
        reranked_results = rerank_results(question, all_results, analysis)
        print(f"Reranked {len(reranked_results)} results")
        print(f"Deduplikowanie: usuwam duplikaty z {len(reranked_results)} wynikow")
        seen: Set[str] = set()
        unique_results = []
        for score, node in reranked_results:
            node_id = node["node"]
            if node_id not in seen:
                unique_results.append((score, node))
                seen.add(node_id)
                if len(unique_results) >= len(embeddings_input) * top_k:
                    break
        print(f"Po deduplikowaniu: {len(unique_results)} unikalnych wynikow")

        question_lower = question.lower()
        is_usage_question = any(word in question_lower for word in ['used', 'where', 'usage', 'called', 'referenced'])

        if is_usage_question:
            print("usage question. Szukam w related_entities")

            target_entity = None
            target_type = None
            if unique_results:
                best_match = unique_results[0]
                node_id = best_match[1]["node"]
                metadata = best_match[1]["metadata"]
                print(f"First result: {node_id}")
                print(f"Label from metadata: {metadata.get('label')}")

                if metadata.get("kind") == "METHOD":
                    target_entity = metadata.get("label")
                    target_type = "method"
                    print(f"Target method identified: {target_entity}")
                elif metadata.get("kind") == "CLASS":
                    target_entity = metadata.get("label")
                    target_type = "class"
                    print(f"Target class identified: {target_entity}")
                elif "." in node_id:
                    parts = node_id.split('.')
                    for part in reversed(parts):
                        if part and part[0].isupper():
                            target_entity = part
                            target_type = "class"
                            print(f"Target class identified from node_id: {target_entity}")
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

                print(f"Added {len(usage_nodes)} usage nodes to results")

            top_nodes = top_nodes[:max_top_nodes_for_usage]


        elif any(word in question_lower for word in ['describe', 'how', 'what']):
            top_nodes = unique_results[:min(max_describe_nodes, len(unique_results))]
        else:
            top_nodes = unique_results[:len(embeddings_input)]

        print(f"Wybrano {len(top_nodes)} najlepszych wezlow")


        category = analysis.get("category", "general")
        confidence = analysis.get("confidence", 0.5)

        print(f"Building context with category={category}, confidence={confidence}")
        full_context = build_context(top_nodes, category, confidence)

        print(f"Context built: {len(full_context)} chars")

        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        print(f"UKONCZONO w :{elapsed_ms:.1f}ms")
        return top_nodes, full_context or "<NO CONTEXT FOUND>"

    except Exception as e:
        print(f"Fallback do oryginalnej funkcji: {e}")
        from graph.retriver import similar_node
        return similar_node(question, model_name, top_k)




_general_fallback_cache: Optional[str] = None
_cache_timestamp: float = 0

