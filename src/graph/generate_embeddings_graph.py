import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.preprocessing import normalize


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Applies mean pooling to token embeddings using an attention mask.

    Aggregates token embeddings by averaging only over non-masked tokens.

    Args:
        token_embeddings (torch.Tensor): Output embeddings from the transformer model.
        attention_mask (torch.Tensor): Binary mask indicating valid tokens (1 = keep).

    Returns:
        torch.Tensor: Mean-pooled sentence embeddings.
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def generate_embeddings_graph(texts: List[str], model_name: str, batch_size: int = 2) -> np.ndarray:
    """
    Generates normalized embeddings for a list of text inputs using CodeBERT.

    Loads the CodeBERT model and tokenizer, encodes the input text in batches,
    applies mean pooling to obtain fixed-size representations, and normalizes
    them using L2 normalization.

    Args:
        texts (List[str]): List of text strings to embed.
        model_name (str): Name of the CodeBERT model to load.
        batch_size (int, optional): Number of samples per batch. Defaults to 2.

    Returns:
        np.ndarray: L2-normalized array of embeddings.
    """
    from src.core.rag_optimization import get_codebert_model

    _codebert_model, _codebert_tokenzier, _device = get_codebert_model()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded_input = _codebert_tokenzier(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            encoded_input = {k: v.to(_device) for k, v in encoded_input.items()}
            model_output = _codebert_model(**encoded_input)
            batch_embeddings = mean_pooling(
                model_output.last_hidden_state, encoded_input["attention_mask"]
            )
            embeddings.extend(batch_embeddings.cpu().numpy())

    return normalize(embeddings, norm="l2")


def extract_code_block_from_file(file_path: Path, location: str) -> str:
    """
    Extracts a code block (e.g., class or method) from a source file.

    Starts reading at the specified line and captures code enclosed by
    matching braces `{}`. Intended for extracting structured code elements.

    Args:
        uri (str): Path to the source file.
        location (str): String containing start position info, e.g. `"123:4;125:0"`.

    Returns:
        str: Extracted code block as a single-line string, or an error message
        if extraction fails.
    """
    try:
        start, _ = location.split(";")
        start_line, _ = map(int, start.split(":"))

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if start_line > len(lines):
            return f"<Invalid start line: {start_line}>"

        block_lines = []
        open_braces = 0
        started = False

        for i in range(start_line - 1, len(lines)):
            line = lines[i]
            block_lines.append(line.rstrip())
            open_braces += line.count("{") - line.count("}")
            if not started and open_braces > 0:
                started = True
            elif started and open_braces == 0:
                break

        code = " ".join(block_lines)
        return re.sub(r"\s+", " ", code).strip()

    except Exception as e:
        return f"<Could not extract code block: {e}>"


def extract_code_from_file(file_path: Path, location: str) -> str:
    """
    Extracts code from a given file starting at a specified line.

    Reads all lines from the given start position until the end of the file.
    Intended for non-structured elements such as variables or constants.

    Args:
        uri (str): Path to the source file.
        location (str): String containing start position info, e.g. `"123:4;125:0"`.

    Returns:
        str: Extracted code text as a single-line string, or an error message
        if reading fails.
    """
    try:
        start, _ = location.split(";")
        start_line, _ = map(int, start.split(":"))

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if start_line > len(lines):
            return f"<Invalid start line: {start_line}>"

        code_lines = [line.rstrip() for line in lines[start_line - 1 :]]
        code = " ".join(code_lines)
        return re.sub(r"\s+", " ", code).strip()

    except Exception as e:
        return f"<Could not extract code: {e}>"


def node_to_text(data: Dict[str, Any], project_root: Path) -> Dict[str, str]:
    """
    Converts a graph node's metadata into textual representation.

    Combines node attributes such as kind and label into a readable text
    and includes the corresponding code snippet extracted from the file.

    Args:
        data (Dict[str, Any]): Node metadata containing fields like
            `kind`, `label`, `uri`, and `location`.

    Returns:
        Dict[str, str]: Dictionary containing:
            - text (str): Human-readable description of the node.
            - kind (str): Type of code element (e.g., CLASS, METHOD).
            - label (str): Node label or identifier.
            - code (str): Extracted code snippet or block.
    """
    label = data.get("label", "")
    kind = data.get("kind", "")
    uri = data.get("uri", "")
    location = data.get("location", "")

    file_path = Path(uri)
    if not file_path.is_absolute():
        file_path = project_root / file_path

    code = (
        extract_code_block_from_file(file_path, location)
        if kind in ["CLASS", "METHOD"]
        else extract_code_from_file(file_path, location)
    )

    return {"text": f"{kind} {label}", "kind": kind, "label": label, "code": code}
