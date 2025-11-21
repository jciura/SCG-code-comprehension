from typing import Any, Dict

MIN_CODE_LENGTH = 30


def get_node_priority_score(node_data: Dict[str, Any], category: str) -> float:
    """
    Computes priority score for a graph node.

    Combines structural importance metrics based on the node type, category,
    and code content (e.g., testing, usage, exception).

    Args:
        node_data: Node data containing 'metadata', 'code', and 'node'
        category: Detected question category (e.g., "usage", "definition")

    Returns:
        Calculated node priority score
    """
    metadata = node_data.get("metadata", {})
    code = node_data.get("code", "")
    node_id = node_data.get("node", "")
    kind = metadata.get("kind", "")

    score = 0.0
    importance = metadata.get("importance", {})
    if isinstance(importance, dict):
        score += importance.get("combined", 0.0) * 0.3
        score += importance.get("pagerank", 0.0) * 100
        score += importance.get("in_degree", 0.0) * 0.1

    if category == "usage":
        if "controller" in node_id.lower() and "test" not in node_id.lower():
            score += 10
        elif "service" in node_id.lower():
            score += 8
        elif "@Mapping" in code or "@RequestMapping" in code:
            score += 12
        elif "repository" in node_id.lower():
            score += 6
    elif category == "definition":
        if kind == "CLASS":
            score += 10
        elif kind == "INTERFACE":
            score += 8
        elif kind == "CONSTRUCTOR":
            score += 6
        if "public class" in code:
            score += 5
    elif category == "implementation":
        if kind == "METHOD" and len(code) > 200:
            score += 8
        elif kind == "CONSTRUCTOR":
            score += 6
        elif "abstract" in code:
            score += 5
    elif category == "testing":
        if "test" in node_id.lower():
            score += 15
        elif "@Test" in code:
            score += 12
        elif "mock" in code.lower():
            score += 6
    elif category == "exception":
        if "exception" in node_id.lower() or "error" in node_id.lower():
            score += 12
        elif "throw" in code.lower() or "catch" in code.lower():
            score += 8

    if len(code) > 500:
        score += 2
    elif len(code) < MIN_CODE_LENGTH:
        score -= 3

    if kind in ["PARAMETER", "VARIABLE", "IMPORT"] and category not in ["definition"]:
        score -= 5

    return score


def get_max_sections_for_category(category: str) -> int:
    """
    Returns the maximum number of code sections to include per category.

    Args:
        category: Question category (e.g., "testing", "usage")

    Returns:
        Maximum number of sections allowed for the given category
    """
    return {
        "testing": 8,
        "usage": 5,
        "implementation": 3,
        "exception": 3,
        "definition": 2,
        "general": 4,
    }.get(category, 2)
