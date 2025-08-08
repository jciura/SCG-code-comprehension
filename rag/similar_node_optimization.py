import time
from functools import lru_cache

_graph_model = None 


def get_graph_model():
    global _graph_model
    if _graph_model is None:
        print("Ładuje model grafu")
        from graph.generate_embeddings_graph import generate_embeddings_graph
        print("Model grafu jest gotowy")
    return True


@lru_cache(maxsize=1000)
def cached_question_question(question_hash):
    from rag.retriver import classify_question, preprocess_question
    return classify_question(preprocess_question(question_hash))


def similar_node_fast(question, model_name="microsoft/codebert-base",top_k=7):

    start_time = time.time()
    try: 
        from rag.retriver import chroma_client, extract_key_value_pairs_simple, classify_question, preprocess_question
        from graph.generate_embeddings_graph import generate_embeddings_graph
        
        collection = chroma_client.get_collection(name="scg_embeddings")
        pairs = extract_key_value_pairs_simple(question)
        embeddings_input = []
        
        for key, value in pairs:
            embeddings_input.append(f"{key} {value}" if key else value)

        if not embeddings_input:
            embeddings_input = [question]
        
        get_graph_model() # upewnienie sie ze model zaladowany
        query_embeddings = generate_embeddings_graph(embeddings_input, model_name)
        all_results = []

        if len(query_embeddings) == 1:
            print("Proste pytanie 1 embedding = 1 zapytanie do chromaDB")
            query_result = collection.query(
                query_embeddings=[query_embeddings[0].tolist()],
                n_results=top_k*len(embeddings_input),
                include=["embeddings", "metadatas", "documents", "distances"]
            )

            for i in range(len(query_result["ids"][0])):
                score = 1 - query_result["distances"][0][i]
                node_id = query_result["ids"][0][i]
                metadata = query_result["metadatas"][0][i]
                code = query_result["documents"][0][i]
                all_results.append((score, {
                    "node": node_id,
                    "metadata": metadata,
                    "code": code
                }))
        else:
            print(f"Złożone pytanie: {len(query_embeddings)} embeddingow = {len(query_embeddings)} zapytan do ChromaDB")

            batch_embeddings = [emb.tolist() for emb in query_embeddings]

            for i,emb in enumerate(batch_embeddings):
                print (f"  Zapytanie {i+1}/{len(batch_embeddings)}: '{embeddings_input[i]}'")
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
                    all_results.append((score, {
                        "node": node_id,
                        "metadata": metadata,
                        "code": code
                    }))

        print(f" Zebrano lacznie {len(all_results)} wynikow z ChromaDb")
        print(f" Deduplikowanie: usuwam duplikaty z {len(all_results)} wynikow")

        seen = set()
        unique_results = []
        for score, node in sorted(all_results, key=lambda x: -x[0]):
            node_id = node["node"]
            if node_id not in seen:
                unique_results.append((score, node))
                seen.add(node_id)
                if len(unique_results) >= len(embeddings_input) * top_k:
                    break
        print(f" Po deduplikowaniu: {len(unique_results)} unikalnych wynikow")


        top_nodes = unique_results[:len(embeddings_input)]
        top_k_codes = [node["code"] for _, node in top_nodes if node["code"]]

        print (f" Wybrano {len(top_nodes)} najlepszych wezlow")

        question_hash = hash(question)
        category = classify_question(preprocess_question(question))
        max_neighbors = {"general": 5, "medium": 3, "specific": 1}.get(category, 2)


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
            print(" Fallback dla general questions - uzywam CACHE ")
            full_context = get_general_fallback_context(collection)
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        print(f" UKONCZONO w :{elapsed_ms:.1f}ms")
        return top_nodes, full_context or "<NO CONTEXT FOUND>"

    except Exception as e:
        print("Fallback do oryginalnej funkcji")
        from rag.retriver import similar_node
        return similar_node(question,model_name, top_k)




_general_fallback_cache = None
_cache_timestamp = 0


def get_general_fallback_context(collection, max_age_seconds=300):

    global _general_fallback_cache, _cache_timestamp

    current_time = time.time()

    if (_general_fallback_cache is None or current_time - _cache_timestamp > max_age_seconds):
        print("Refreshing cache - (nie laduje wszystkich danych)")
        try:
            all_nodes = collection.get(include=["metadatas", "ids"], limit=100)
            importance_scores = []
            for i in range(len(all_nodes["ids"])):
                meta = all_nodes["metadatas"][i]
                nid = all_nodes["ids"][i]
                score = meta.get("importance", {}).get("combined", 0.0)
                importance_scores.append((score, nid))
            sorted_by_importance = sorted(importance_scores, key=lambda x: -x[0])
            top_ids = [nid for _, nid in sorted_by_importance[:5]]

            top_nodes = collection.get(
                ids=top_ids,
                include=['documents'] # pobieram kontent tylko dla wybranych dokumentow
            )
            fallback_docs = [doc for doc in top_nodes["documents"] if doc]
            _general_fallback_cache = "\n\n".join(fallback_docs)
            _cache_timestamp = current_time

            print(f"Cashed {len(fallback_docs)} fallback documents")
            
        except Exception as e:
            print(f"Error podczas fallbacku cache {e}")
            _general_fallback_cache = "<NO CONTEXT FOUND>" 
    
    return _general_fallback_cache