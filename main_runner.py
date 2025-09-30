import json
import os
from collections import defaultdict
from chroma_client import get_chroma_client, get_or_create_collection
from graph.load_graph import load_gdf, extract_scores
from graph.generate_embeddings_graph import generate_embeddings_graph, node_to_text
from graph.question_embedding import generate_and_save_classifier_embeddings


def load_graph_main() -> None:
    load_gdf("../projects/test.gdf")
    extract_scores("../projects/partition.js")


def question_embedding_main() -> None:
    generate_and_save_classifier_embeddings("../embeddings/classifier_example_embeddings.json")


def generate_embeddings_graph_main() -> None:
    model_name = "microsoft/codebert-base"
    scg = load_gdf('../projects/scgTest.gdf')
    ccn = load_gdf('../projects/ccnTest.gdf')
    importance_scores = extract_scores("../projects/partition.js")
    chroma_storage_path = "embeddings/chroma_storage"
    collection_name = "scg_embeddings"

    reverse_ccn_map = defaultdict(list)
    for node_id in ccn.nodes():
        for neighbor in ccn.neighbors(node_id):
            reverse_ccn_map[neighbor].append(node_id)

    nodes_info = []
    texts_for_embedding = []

    for node_id, data in scg.nodes(data=True):
        node_text = node_to_text(data)
        nodes_info.append({
            "node_id": node_id,
            "kind": node_text["kind"],
            "label": node_text["label"],
            "code": node_text["code"]
        })
        texts_for_embedding.append(node_text["text"].lower())

    embeddings = generate_embeddings_graph(texts_for_embedding, model_name)
    collection = get_or_create_collection(
        collection_name=collection_name,
        storage_path=chroma_storage_path
    )
    json_data = []

    for info, emb in zip(nodes_info, embeddings):
        node_id = info["node_id"]

        scg_neighbors = set(scg.neighbors(node_id)) if scg.has_node(node_id) else set()
        used_by = set(reverse_ccn_map[node_id]) if node_id in reverse_ccn_map else set()

        extra_related = set()
        if info["kind"] == "METHOD":
            class_id = node_id.split('(')[0].rsplit('.', 1)[0]
            if class_id in reverse_ccn_map:
                extra_related.update(reverse_ccn_map[class_id])

        related_entities = sorted(
            scg_neighbors.union(used_by).union(extra_related),
            key=lambda nid: importance_scores["combined"].get(nid, 0.0),
            reverse=True
        )

        metadata = {
            "node": node_id,
            "kind": info["kind"],
            "label": info["label"],
            "related_entities": json.dumps(related_entities),
            "loc": importance_scores['loc'].get(node_id, 0.0),
            "out_degree": importance_scores['out-degree'].get(node_id, 0.0),
            "in_degree": importance_scores['in-degree'].get(node_id, 0.0),
            "pagerank": importance_scores['pagerank'].get(node_id, 0.0),
            "eigenvector": importance_scores['eigenvector'].get(node_id, 0.0),
            "katz": importance_scores['Katz'].get(node_id, 0.0),
            "combined": importance_scores['combined'].get(node_id, 0.0),
        }

        json_data.append({
            **metadata,
            "code": info["code"],
            "embedding": emb.tolist()
        })

        try:
            collection.add(
                ids=[node_id],
                embeddings=[emb.tolist()],
                metadatas=[metadata],
                documents=[info["code"]]
            )
        except Exception as e:
            print(f"Failed to add {node_id}: {str(e)}")
    # with open("../embeddings/node_embedding.json", "w", encoding="utf-8") as f:
    #     json.dump(json_data, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    load_graph_main()
    question_embedding_main()
    generate_embeddings_graph_main()
