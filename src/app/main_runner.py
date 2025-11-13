import json
from collections import defaultdict
from src.clients.chroma_client import get_or_create_collection
from src.graph.load_graph import load_gdf, extract_scores
from src.graph.generate_embeddings_graph import generate_embeddings_graph, node_to_text
from src.core.config import CODEBERT_MODEL_NAME, default_chroma_path, default_collection_name, partition, test_gdf, scg_test, ccn_test
from loguru import logger

def load_graph_main() -> None:
    """
        Loads and processes the primary test graph data.

        Loads a graph from the configured test GDF file and extracts node importance
        scores using the specified partition configuration.

        Returns:
            None
    """
    load_gdf(test_gdf)
    extract_scores(partition)



def generate_embeddings_graph_main() -> None:
    """
        Generates graph node embeddings and stores them in a Chroma collection.

        Loads SCG and CCN graphs, computes importance scores, builds reverse mappings
        for node relationships, converts nodes to textual representations, and generates
        embeddings using the configured CodeBERT model.
        Each node and its metadata (including structural metrics and related entities)
        are then stored in the Chroma vector database.

        Returns:
            None

        Raises:
            Exception: If embedding generation or data insertion into the Chroma collection fails.
    """
    scg = load_gdf(scg_test)
    ccn = load_gdf(ccn_test)
    importance_scores = extract_scores(partition)

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

    embeddings = generate_embeddings_graph(texts_for_embedding, CODEBERT_MODEL_NAME)
    collection = get_or_create_collection(
        collection_name=default_collection_name,
        storage_path=default_chroma_path
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
            logger.error(f"Failed to add {node_id}: {str(e)}")
    # with open("../embeddings/node_embedding.json", "w", encoding="utf-8") as f:
    #     json.dump(json_data, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    load_graph_main()
    generate_embeddings_graph_main()
