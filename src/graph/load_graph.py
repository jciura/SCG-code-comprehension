import json
from typing import Any, Dict

import networkx as nx


def load_gdf(filepath: str) -> nx.DiGraph:
    """
    Loads a directed graph from a `.gdf` file.

    Parses node and edge definitions to build a NetworkX directed graph,
    preserving attributes defined in the file.

    Args:
        filepath (str): Path to the `.gdf` graph file.

    Returns:
        nx.DiGraph: Directed graph containing all nodes and edges
        with associated attributes.
    """
    G = nx.DiGraph()
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    node_section = False
    edge_section = False
    node_attrs = []
    edge_attrs = []
    for line in lines:
        line = line.strip()
        if line.startswith("nodedef>"):
            node_section = True
            edge_section = False
            node_attrs = line[len("nodedef>") :].split(",")
            node_attrs = [a.strip().split(" ")[0] for a in node_attrs]
            continue
        if line.startswith("edgedef>"):
            edge_section = True
            node_section = False
            edge_attrs = line[len("edgedef>") :].split(",")
            edge_attrs = [a.strip().split(" ")[0] for a in edge_attrs]
            continue

        if node_section:
            if line == "" or line.startswith("#"):
                continue
            values = line.split(",")
            node_id = values[0].strip()
            attr_dict = {node_attrs[i]: values[i].strip() for i in range(1, len(values))}
            G.add_node(node_id, **attr_dict)

        if edge_section:
            if line == "" or line.startswith("#"):
                continue
            values = line.split(",")
            source = values[0].strip()
            target = values[1].strip()
            attr_dict = {edge_attrs[i]: values[i].strip() for i in range(2, len(values))}
            G.add_edge(source, target, **attr_dict)

    return G


def load_crucial_from_js(js_path: str) -> Dict[str, Any]:
    """
    Parses and loads JSON-like data from a `.js` file created by `scg-cli`.

    Extracts the JSON object assigned to a variable (e.g., `const crucial = {...};`)
    and converts it into a Python dictionary.

    Args:
        js_path (str): Path to the JavaScript file containing JSON data.

    Returns:
        Dict[str, Any]: Parsed JSON data as a Python dictionary.
    """
    with open(js_path, "r", encoding="utf-8") as f:
        content = f.read()

    start = content.find("{")
    end = content.rfind("}")
    json_str = content[start : end + 1]

    data = json.loads(json_str)
    return data


def extract_scores(js_path: str) -> Dict[str, Dict[str, float]]:
    """
    Extracts node importance scores from a parsed `crucial.js` or `partition.js` file.

    Reads statistical metrics (e.g., PageRank, eigenvector, Katz) for all nodes
    and returns a nested dictionary mapping metric names to node-score pairs.

    Args:
        js_path (str): Path to the JavaScript file containing metric data.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary where keys are metric IDs and
        values are dictionaries of node IDs with their associated scores.
    """
    data = load_crucial_from_js(js_path)
    scores = {}

    for stat in data["stats"]:
        metric_id = stat["id"]
        scores[metric_id] = {}
        for node_info in stat["nodes"]:
            node_id = node_info["id"]
            node_score = node_info["score"]
            scores[metric_id][node_id] = node_score

    return scores
