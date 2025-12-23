import json
import re
import shutil
import subprocess
from pathlib import Path

from loguru import logger

from src.clients.chroma_client import get_or_create_collection
from src.core.config import (
    CODEBERT_MODEL_NAME,
    default_chroma_path,
    default_collection_name,
    partition,
    scg_test,
)
from src.graph.generate_embeddings_graph import generate_embeddings_graph, node_to_text
from src.graph.load_graph import extract_scores, load_gdf


def run_scg_cli(project_path: Path, output_folder: Path):
    output_folder.mkdir(parents=True, exist_ok=True)
    project_parent = project_path.parent
    project_name = project_path.name

    # gen_cmd = ["scg-cli", "generate", str(project_path)]
    # logger.info(f"Running: {' '.join(gen_cmd)}")
    # subprocess.run(gen_cmd, check=True, cwd=project_path, shell=True)

    ccn_cmd = ["scg-cli", "export", "-g", "CCN", "-o", "gdf", str(project_path)]
    logger.info(f"Running: {' '.join(ccn_cmd)}")
    subprocess.run(ccn_cmd, check=True, cwd=project_path, shell=True)

    generated_ccn_file = project_parent / f"{project_name}.gdf"
    moved_ccn = output_folder / "ccnTest.gdf"
    if generated_ccn_file.exists():
        shutil.move(str(generated_ccn_file), str(moved_ccn))
        logger.info(f"CCN graph moved to {moved_ccn}")
    else:
        logger.error(f"CCN graph not found at {moved_ccn}")

    scg_cmd = ["scg-cli", "export", "-g", "SCG", "-o", "gdf", str(project_path)]
    logger.info(f"Running: {' '.join(scg_cmd)}")
    subprocess.run(scg_cmd, check=True, cwd=project_path, shell=True)

    generated_scg_file = project_parent / f"{project_name}.gdf"
    moved_scg = output_folder / "scgTest.gdf"
    if generated_scg_file.exists():
        shutil.move(str(generated_scg_file), str(moved_scg))
        logger.info(f"SCG graph moved to {moved_scg}")
    else:
        logger.error(f"SCG graph not found at {moved_scg}")

    crucial_cmd = ["scg-cli", "crucial", str(project_path)]
    logger.info(f"Running: {' '.join(crucial_cmd)}")
    subprocess.run(crucial_cmd, check=True, cwd=project_path.parent, shell=True)

    generated_crucial_file = project_parent / "crucial.html"
    generated_partition_file = project_parent / "partition.js"
    moved_crucial = output_folder / "crucial.html"
    moved_partition = output_folder / "partition.js"
    if generated_crucial_file.exists():
        shutil.move(str(generated_crucial_file), str(moved_crucial))
        logger.info(f"Crucial nodes file moved to {moved_crucial}")
    else:
        logger.error(f"Crucial nodes file not found at {moved_crucial}")

    if generated_partition_file.exists():
        shutil.move(str(generated_partition_file), str(moved_partition))
        logger.info(f"Partition file moved to {moved_partition}")
    else:
        logger.error(f"Partition file not found at {moved_partition}")


def load_graph_main() -> None:
    """
    Loads and processes the primary test graph data.

    Loads a graph from the configured test GDF file and extracts node importance
    scores using the specified partition configuration.

    Returns:
        None
    """
    load_gdf(scg_test)
    extract_scores(partition)


def generate_embeddings_graph_main(project_path: Path) -> None:
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
    importance_scores = extract_scores(partition)

    nodes_info = []
    texts_for_embedding = []

    for node_id, data in scg.nodes(data=True):
        node_text = node_to_text(data, project_path)
        nodes_info.append(
            {
                "node_id": node_id,
                "kind": node_text["kind"],
                "label": node_text["label"],
                "code": node_text["code"],
                "uri": node_text["uri"]
            }
        )
        texts_for_embedding.append(node_text["text"].lower())

    embeddings = generate_embeddings_graph(texts_for_embedding, CODEBERT_MODEL_NAME)
    collection = get_or_create_collection(
        collection_name=default_collection_name, storage_path=default_chroma_path
    )
    json_data = []

    max_combined = 0
    for info, emb in zip(nodes_info, embeddings):
        node_id = info["node_id"]

        related_entities = []
        for neighbor_id in set(scg.successors(node_id)) | set(scg.predecessors(node_id)):
            if scg.has_edge(node_id, neighbor_id):
                type = scg[node_id][neighbor_id].get("type", "")
                related_entities.append([neighbor_id, type])

            if scg.has_edge(neighbor_id, node_id):
                type = scg[neighbor_id][node_id].get("type", "")
                related_entities.append([neighbor_id, type + "_BY"])

        related_entities = sorted(
            related_entities,
            key=lambda x: importance_scores["combined"].get(x[0], 0.0),
            reverse=True,
        )

        metadata = {
            "node": node_id,
            "kind": info["kind"].upper(),
            "label": info["label"],
            "uri": info["uri"],
            "related_entities": json.dumps(related_entities),
            "loc": importance_scores["loc"].get(node_id, 0.0),
            "out_degree": importance_scores["out-degree"].get(node_id, 0.0),
            "in_degree": importance_scores["in-degree"].get(node_id, 0.0),
            "pagerank": importance_scores["pagerank"].get(node_id, 0.0),
            "eigenvector": importance_scores["eigenvector"].get(node_id, 0.0),
            "katz": importance_scores["Katz"].get(node_id, 0.0),
            "combined": importance_scores["combined"].get(node_id, 0.0),
        }

        node_combined = importance_scores["combined"].get(node_id, 0.0)
        if node_combined > max_combined:
            max_combined = node_combined

        json_data.append({**metadata, "code": info["code"], "embedding": emb.tolist()})

        try:
            collection.add(
                ids=[node_id],
                embeddings=[emb.tolist()],
                metadatas=[metadata],
                documents=[info["code"]],
            )
        except Exception as e:
            logger.error(f"Failed to add {node_id}: {str(e)}")
    with open("../../data/embeddings/node_embedding.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    with open("../core/config.py", "r") as f:
        content = f.read()

    new_max_combined = re.sub(r'COMBINED_MAX\s*=\s*\d+', f'COMBINED_MAX = {max_combined}', content)

    with open("../core/config.py", "w") as f:
        f.write(new_max_combined)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate SCG embeddings for project")
    parser.add_argument("--project", type=str, required=True, help="Path to the project folder")
    args = parser.parse_args()

    project_path = Path(args.project).resolve()
    program_graph_folder = Path(__file__).parent.parent.parent / "data/graph"
    run_scg_cli(project_path, program_graph_folder)
    generate_embeddings_graph_main(project_path)
