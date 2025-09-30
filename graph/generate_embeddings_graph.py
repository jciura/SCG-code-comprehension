import os
import re
from typing import List, Dict, Any
import torch
from sklearn.preprocessing import normalize
import numpy as np


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / \
        torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def generate_embeddings_graph(texts: List[str], model_name: str, batch_size: int = 2) -> np.ndarray:
    from rag_optimization import get_codebert_model
    _codebert_model, _codebert_tokenzier, _device = get_codebert_model()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded_input = _codebert_tokenzier(
                batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
            encoded_input = {k: v.to(_device) for k, v in encoded_input.items()}
            model_output = _codebert_model(**encoded_input)
            batch_embeddings = mean_pooling(model_output.last_hidden_state, encoded_input['attention_mask'])
            embeddings.extend(batch_embeddings.cpu().numpy())

    return normalize(embeddings, norm='l2')


def extract_code_block_from_file(uri: str, location: str) -> str:
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


def extract_code_from_file(uri: str, location: str) -> str:
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


def node_to_text(data: Dict[str, Any]) -> Dict[str, str]:
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
