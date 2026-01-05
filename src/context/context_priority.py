from typing import Any, Dict, Optional

MIN_CODE_LENGTH = 100
CODE_BONUS_MIN_LENGTH = 500
IMPLEMENTATION_MIN_LENGTH = 200


def get_node_priority_score(
        node_data: Dict[str, Any],
        category: str,
        target: Optional[str] = None
) -> float:
    """
    Computes priority score for a graph node.

    Args:
        node_data: Node data containing 'metadata', 'code', and 'node'
        category: Detected question category (e.g., "usage", "definition")
        target: Target entity name for usage category

    Returns:
        Calculated node priority score
    """
    metadata = node_data.get("metadata", {})
    code = node_data.get("code", "")
    node_id = node_data.get("node", "")
    kind = metadata.get("kind", "")
    node_id_lower = node_id.lower()
    code_lower = code.lower()
    target_lower = target.lower() if target else ""

    score = 0.0
    score += float(metadata.get("combined", 0.0)) * 0.3
    score += float(metadata.get("pagerank", 0.0)) * 100
    score += float(metadata.get("in_degree", 0.0)) * 0.1

    if category == "usage":
        if "test" in node_id_lower or "spec" in node_id_lower:
            score += 15
        elif target_lower:
            if f"{target_lower}(" in code_lower or f"{target_lower} (" in code_lower:
                score += 12
            elif f"new {target_lower}" in code_lower:
                score += 12
            elif f".{target_lower}(" in code_lower:
                score += 10
            elif any(p in code_lower for p in [
                f"extends {target_lower}",
                f"with {target_lower}",
                f"implements {target_lower}",
                f"<: {target_lower}",]):
                score += 10
            elif f": {target_lower}" in code_lower or f"[{target_lower}]" in code_lower:
                score += 8
            elif f"<{target_lower}>" in code_lower:
                score += 8
            elif f"{target_lower} " in code_lower:
                score += 6
            elif "implicit" in code_lower and target_lower in code_lower:
                score += 10
        if kind == "METHOD":
            score += 3
    elif category == "definition":
        if kind in ["CLASS", "TRAIT", "OBJECT"]:
            score += 10
        elif kind == "INTERFACE":
            score += 8
        elif kind == "CONSTRUCTOR":
            score += 6
        if "public class" in code_lower or "case class" in code_lower:
            score += 5
    elif category == "implementation":
        if kind == "METHOD" and len(code) > IMPLEMENTATION_MIN_LENGTH:
            score += 8
        elif kind in ["CONSTRUCTOR", "ENUM"]:
            score += 6
        elif "abstract" in code_lower or "override" in code_lower:
            score += 5
    elif category == "testing":
        if "test" in node_id_lower or "spec" in node_id_lower:
            score += 15
        elif "@test" in code_lower:
            score += 12
        elif "mock" in code_lower or "stub" in code_lower:
            score += 6
        elif any(kw in code_lower for kw in ["assert", "should", "must", "expect"]):
            score += 4
    elif category == "exception":
        if "exception" in node_id_lower or "error" in node_id_lower:
            score += 12
        elif "throw" in code_lower or "catch" in code_lower:
            score += 8
        elif "recover" in code_lower:
            score += 6

    if len(code) > CODE_BONUS_MIN_LENGTH:
        score += 2
    elif len(code) < MIN_CODE_LENGTH:
        score -= 3

    if kind in ["PARAMETER", "VARIABLE", "IMPORT"] and category != "definition":
        score -= 5

    return score


def get_max_sections_for_category(category: str) -> int:
    """
    Returns the maximum number of code sections to include per category.
    """
    return {
        "testing": 8,
        "usage": 5,
        "implementation": 3,
        "exception": 3,
        "definition": 2,
        "general": 4,
    }.get(category, 2)