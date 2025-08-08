import json
import os
import re
from collections import defaultdict

import chromadb
import torch
from chromadb.config import Settings
from sklearn.preprocessing import normalize


from graph.load_graph import load_gdf, extract_scores

os.makedirs(os.path.abspath("../embeddings/chroma_storage"), exist_ok=True)
chroma_client = chromadb.PersistentClient(
    path="../embeddings/chroma_storage",
    settings=Settings(allow_reset=False)
)


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / \
        torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def generate_embeddings_graph(texts, model_name, batch_size=2):
    from rag_optimization import get_codebert_model
    _codebert_model, _codebert_tokenzier, _device = get_codebert_model()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded_input = _codebert_tokenzier(
                batch, padding=True, truncation=True, max_length=512, return_tensors='pt'
            )
            encoded_input = {k: v.to(_device) for k, v in encoded_input.items()}
            model_output = _codebert_model(**encoded_input)
            batch_embeddings = mean_pooling(model_output.last_hidden_state, encoded_input['attention_mask'])
            embeddings.extend(batch_embeddings.cpu().numpy())

    return normalize(embeddings, norm='l2')


def extract_code_block_from_file(uri, location):
    try:
        start, _ = location.split(';')
        start_line, _ = map(int, start.split(':'))

        with open(uri, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if start_line > len(lines):
            return f"<Invalid start line: {start_line}>"

        block_lines = []
        open_braces = 0
        started = False

        for i in range(start_line - 1, len(lines)):
            line = lines[i]
            block_lines.append(line.rstrip())
            open_braces += line.count('{') - line.count('}')
            if not started and open_braces > 0:
                started = True
            elif started and open_braces == 0:
                break

        code = ' '.join(block_lines)
        return re.sub(r'\s+', ' ', code).strip()

    except Exception as e:
        return f"<Could not extract code block: {e}>"


def extract_code_from_file(uri, location):
    try:
        start, _ = location.split(';')
        start_line, _ = map(int, start.split(':'))

        with open(uri, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if start_line > len(lines):
            return f"<Invalid start line: {start_line}>"

        code_lines = [line.rstrip() for line in lines[start_line - 1:]]
        code = ' '.join(code_lines)
        return re.sub(r'\s+', ' ', code).strip()

    except Exception as e:
        return f"<Could not extract code: {e}>"


def node_to_text(data):
    label = data.get('label', '')
    kind = data.get('kind', '')
    uri = data.get('uri', '')
    location = data.get('location', '')

    code = (
        extract_code_block_from_file(uri, location)
        if kind in ['CLASS', 'METHOD']
        else extract_code_from_file(uri, location)
    )

    return {
        "text": f"{kind} {label}",
        "kind": kind,
        "label": label,
        "code": code
    }


if __name__ == "__main__":
    MODEL_NAME = "microsoft/codebert-base"
    scg = load_gdf('../projects/scgTest.gdf')
    ccn = load_gdf('../projects/ccnTest.gdf')
    importance_scores = extract_scores("../projects/partition.js")

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

    embeddings = generate_embeddings_graph(texts_for_embedding, MODEL_NAME)
    collection = chroma_client.get_or_create_collection(name="scg_embeddings")

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
