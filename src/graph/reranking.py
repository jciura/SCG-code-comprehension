import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from src.core.intent_analyzer import IntentCategory


def extract_target_class_name(query: str) -> Optional[str]:
    """
    Extracts target class name for testing queries.

    Args:
        query: User query string

    Returns:
        Extracted class name or None if not found
    """
    class_match = re.search(r"for\s+(\w+)\s+class", query.lower())
    if class_match:
        return class_match.group(1).lower()

    class_patterns = [
        r"(\w+controller)\s+class",
        r"(\w+service)\s+class",
        r"(\w+repository)\s+class",
        r"tests?\s+for\s+(\w+)",
        r"(\w+)\s+tests?",
        r"test.*(\w+controller)",
        r"test.*(\w+service)",
    ]
    for pattern in class_patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1).lower()
    return None


def testing_boost(
    adjusted_score: float,
    node_id: str,
    metadata: Dict[str, Any],
    code: str,
    target_class_name: Optional[str],
) -> float:
    """
    Applies testing-specific score boosts.

    Args:
        adjusted_score: Base score to adjust
        node_id: Node identifier
        metadata: Node metadata
        code: Node code content
        target_class_name: Target class name for testing queries (used for boosting)

    Returns:
        Adjusted score with testing boosts applied
    """
    kind = metadata.get("kind", "")
    label = metadata.get("label", "")

    if target_class_name and target_class_name in node_id.lower():
        if kind == "METHOD" and "test" in node_id.lower():
            adjusted_score *= 10.0
            logger.debug(f"boost for test method: {node_id} - {adjusted_score:.3f}")
        elif kind == "CLASS" and "test" in node_id.lower():
            adjusted_score *= 8.0
            logger.debug(f"boost for test class: {node_id} - {adjusted_score:.3f}")
        elif kind == "METHOD" and any(
            test_word in label.lower() for test_word in ["should", "test"]
        ):
            adjusted_score *= 7.0
            logger.debug(f"test method boost: {node_id} - {adjusted_score:.3f}")

    if "test" in node_id.lower():
        if kind == "METHOD":
            adjusted_score *= 3.0
        elif kind == "CLASS":
            adjusted_score *= 2.5
    elif "@test" in code.lower():
        adjusted_score *= 2.0
    elif kind == "METHOD" and ("should" in label.lower() or "test" in label.lower()):
        adjusted_score *= 1.8

    if (
        "test" not in node_id.lower()
        and "@test" not in code.lower()
        and not any(test_word in label.lower() for test_word in ["should", "test"])
    ):
        adjusted_score *= 0.2
    return adjusted_score


def usage_boost(adjusted_score: float, node_id: str, metadata: Dict[str, Any], code: str) -> float:
    """
    Applies usage-specific score boosts.

    Args:
        adjusted_score: Base score to adjust
        node_id: Node identifier
        metadata: Node metadata
        code: Node code content

    Returns:
        Adjusted score with usage boosts applied
    """
    kind = metadata.get("kind", "")

    if "controller" in node_id.lower() and "test" not in node_id.lower():
        adjusted_score *= 2.0
    elif "service" in node_id.lower():
        adjusted_score *= 1.5
    elif kind == "METHOD" and any(
        annotation in code
        for annotation in ["@GetMapping", "@PostMapping", "@PutMapping", "@DeleteMapping"]
    ):
        adjusted_score *= 1.8
    elif "repository" in node_id.lower():
        adjusted_score *= 1.3
    return adjusted_score


def definition_boost(adjusted_score: float, metadata: Dict[str, Any], code: str) -> float:
    """
    Applies definition-specific score boosts.

    Args:
        adjusted_score: Base score to adjust
        metadata: Node metadata
        code: Node code content

    Returns:
        Adjusted score with definition boosts applied
    """
    kind = metadata.get("kind", "")
    label = metadata.get("label", "")
    if kind == "CLASS":
        adjusted_score *= 1.8
    elif kind == "INTERFACE":
        adjusted_score *= 1.6
    elif kind == "CONSTRUCTOR":
        adjusted_score *= 1.4
    elif kind == "METHOD" and label.lower() in ["main", "init", "setup"]:
        adjusted_score *= 1.3
    elif kind == "VARIABLE" and "final" in code.lower():
        adjusted_score *= 0.9
    return adjusted_score


def implementation_boost(adjusted_score: float, metadata: Dict[str, Any], code: str) -> float:
    """
    Applies implementation-specific score boosts.

    Args:
        adjusted_score: Base score to adjust
        metadata: Node metadata
        code: Node code content

    Returns:
        Adjusted score with implementation boosts applied
    """
    kind = metadata.get("kind", "")

    if kind == "METHOD":
        adjusted_score *= 1.5
    elif kind == "CONSTRUCTOR":
        adjusted_score *= 1.3
    elif kind == "CLASS" and "abstract" in code.lower():
        adjusted_score *= 1.2
    elif kind == "VARIABLE":
        adjusted_score *= 0.8

    return adjusted_score


def exception_boost(
    adjusted_score: float, node_id: str, metadata: Dict[str, Any], code: str
) -> float:
    """
    Applies exception-specific score boosts.

    Args:
        adjusted_score: Base score to adjust
        node_id: Node identifier
        metadata: Node metadata
        code: Node code content

    Returns:
        Adjusted score with exception boosts applied
    """
    kind = metadata.get("kind", "")
    label = metadata.get("label", "")

    if "exception" in node_id.lower() or "error" in node_id.lower():
        adjusted_score *= 2.0
    elif kind == "CLASS" and ("Exception" in label or "Error" in label):
        adjusted_score *= 1.8
    elif "throw" in code.lower() or "catch" in code.lower():
        adjusted_score *= 1.5

    return adjusted_score


def general_factors(
    adjusted_score: float,
    node_id: str,
    metadata: Dict[str, Any],
    code: str,
    query: str,
    category: str,
    confidence: float,
) -> float:
    """
    Applies general reranking factors (length, importance, overlap, etc.).

    Args:
        adjusted_score: Base score to adjust
        node_id: Node identifier
        metadata: Node metadata
        code: Node code content
        query: User query string
        category: Intent category
        confidence: Confidence score

    Returns:
        Adjusted score with general factors applied
    """
    kind = metadata.get("kind", "")
    label = metadata.get("label", "")

    if len(code) < 100:
        adjusted_score *= 0.7
    elif len(code) > 1000:
        adjusted_score *= 1.1

    importance = metadata.get("importance", {})
    if isinstance(importance, dict):
        combined_score = importance.get("combined", 0.0)
        if combined_score > 10.0:
            adjusted_score *= 1.4
        elif combined_score > 5.0:
            adjusted_score *= 1.2
        elif combined_score > 2.0:
            adjusted_score *= 1.1

        pagerank = importance.get("pagerank", 0.0)
        if pagerank > 0.01:
            adjusted_score *= 1.2
        elif pagerank > 0.005:
            adjusted_score *= 1.1

        in_degree = importance.get("in-degree", 0.0)
        if in_degree > 5:
            adjusted_score *= 1.3
        elif in_degree > 2:
            adjusted_score *= 1.1

    query_terms = set(query.lower().split())
    label_terms = set(label.lower().split())
    node_id_terms = set(
        node_id.lower().replace(".", " ").replace("(", " ").replace(")", " ").split()
    )

    term_overlap = len(query_terms.intersection(label_terms.union(node_id_terms)))
    if term_overlap > 0:
        adjusted_score *= 1.0 + 0.15 * term_overlap
    if kind == "PARAMETER" or kind == "VARIABLE":
        adjusted_score *= 0.6

    if category != "testing" and "test" in node_id.lower():
        adjusted_score *= 0.8
    if confidence > 0.7:
        adjusted_score *= 1.0 + (confidence - 0.7) * 0.5

    return adjusted_score


def rerank_results(
    query: str, nodes: List[Tuple[float, Dict[str, Any]]], analyses: Dict[str, Any]
) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Reranks retrieved nodes based on intent category and various heuristics.

    Args:
        query: Original user query text
        nodes: Retrieved items as (score, node_data)
        analyses: Analysis payload with category and confidence

    Returns:
        Reranked items sorted by adjusted score (descending)
    """
    category = analyses.get("category", "general")
    confidence = analyses.get("confidence", 0.5)

    try:
        intent_category = IntentCategory(category)
    except ValueError:
        intent_category = IntentCategory.GENERAL

    target_class_name = None
    if intent_category == IntentCategory.TESTING:
        target_class_name = extract_target_class_name(query)
        if target_class_name:
            logger.debug(f"Testing category, target class: {target_class_name}")

    reranked = []
    for item in nodes:
        if isinstance(item, tuple) and len(item) == 2:
            score, node_data = item
        else:
            score = item.get("score", 0.5)
            node_data = item

        node_id = node_data.get("node", "")
        metadata = node_data.get("metadata", {})
        code = node_data.get("code", "")

        adjusted_score = float(score)

        if intent_category == IntentCategory.TESTING:
            adjusted_score = testing_boost(
                adjusted_score, node_id, metadata, code, target_class_name
            )
        elif intent_category == IntentCategory.USAGE:
            adjusted_score = usage_boost(adjusted_score, node_id, metadata, code)
        elif intent_category == IntentCategory.DEFINITION:
            adjusted_score = definition_boost(adjusted_score, metadata, code)
        elif intent_category == IntentCategory.IMPLEMENTATION:
            adjusted_score = implementation_boost(adjusted_score, metadata, code)
        elif intent_category == IntentCategory.EXCEPTION:
            adjusted_score = exception_boost(adjusted_score, node_id, metadata, code)

        adjusted_score = general_factors(
            adjusted_score, node_id, metadata, code, query, category, confidence
        )
        reranked.append((adjusted_score, node_data))

    reranked.sort(key=lambda x: x[0], reverse=True)

    if intent_category == IntentCategory.TESTING:
        logger.debug("\nTop 10 reranked results for testing:")
        for i, (score, node_data) in enumerate(reranked[:10]):
            node_id = node_data.get("node", "")
            kind = node_data.get("metadata", {}).get("kind", "")
            logger.debug(f"{i + 1}. {node_id} ({kind}) - Score: {score:.3f}")

    return reranked
