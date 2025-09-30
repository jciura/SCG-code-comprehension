import json

import networkx as nx
from typing import Dict, Any


# Konwerter pliku z rozszerzeniem .gdf (pliku grafu)
def load_gdf(filepath: str) -> nx.DiGraph:
    G = nx.DiGraph()

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    node_section = False
    edge_section = False
    node_attrs = []
    edge_attrs = []

    for line in lines:
        line = line.strip()
        if line.startswith('nodedef>'):
            node_section = True
            edge_section = False
            node_attrs = line[len('nodedef>'):].split(',')
            node_attrs = [a.strip().split(' ')[0] for a in node_attrs]
            continue
        if line.startswith('edgedef>'):
            edge_section = True
            node_section = False
            edge_attrs = line[len('edgedef>'):].split(',')
            edge_attrs = [a.strip().split(' ')[0] for a in edge_attrs]
            continue

        if node_section:
            if line == '' or line.startswith('#'):
                continue
            values = line.split(',')
            node_id = values[0].strip()
            attr_dict = {node_attrs[i]: values[i].strip() for i in range(1, len(values))}
            G.add_node(node_id, **attr_dict)

        if edge_section:
            if line == '' or line.startswith('#'):
                continue
            values = line.split(',')
            source = values[0].strip()
            target = values[1].strip()
            attr_dict = {edge_attrs[i]: values[i].strip() for i in range(2, len(values))}
            G.add_edge(source, target, **attr_dict)

    return G


def load_crucial_from_js(js_path: str) -> Dict[str, Any]:
    with open(js_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Usuń `const crucial =` i końcową średnikę
    start = content.find('{')
    end = content.rfind('}')
    json_str = content[start:end + 1]

    data = json.loads(json_str)
    return data


# Wyciągnięcie wskaźników wagi węzłów z pliku partition.js
# (zrobione poleceniem: scg-cli crucial <Sciezka>)
# scg-cli robi plik partiton.js zamiast crucial.js - trzeba zmienić w tamtym kodzie
def extract_scores(js_path: str) -> Dict[str, Dict[str, float]]:
    data = load_crucial_from_js(js_path)
    scores = {}

    for stat in data['stats']:
        metric_id = stat['id']
        scores[metric_id] = {}
        for node_info in stat['nodes']:
            node_id = node_info['id']
            node_score = node_info['score']
            scores[metric_id][node_id] = node_score

    return scores

