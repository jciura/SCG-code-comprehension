from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from core.models import IntentAnalysis
from src.core.intent_analyzer import IntentCategory
from context.context_extraction import extract_target_from_question

ignored_kinds = frozenset({"PARAMETER", "VARIABLE", "IMPORT", "VALUE"})


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
    if kind in ignored_kinds:
        return adjusted_score * 0.3
    label = metadata.get("label", "").lower()
    node_id_lower = node_id.lower()
    code_lower = code.lower()
    is_test_node = any(key_word in node_id_lower for key_word in ["test", "spec", "suite"])
    has_test_annotation = "@test" in code_lower
    has_scala_test = any(key_word in code_lower for key_word in [
        "shouldbe", "should be", "mustbe", "must be", "assertresult", "intercept["])
    has_test_label = any(key_word in label for key_word in ["should", "test", "must", "expect"])
    is_test_related = is_test_node or has_test_annotation or has_scala_test or has_test_label
    if not is_test_related:
        return adjusted_score * 0.2
    if target_class_name and target_class_name.lower() in node_id_lower:
        if kind == "METHOD" and is_test_node:
            adjusted_score *= 10.0
        elif kind == "CLASS" and is_test_node:
            adjusted_score *= 8.0
        elif kind == "METHOD" and has_test_label:
            adjusted_score *= 7.0
    elif is_test_node:
        if kind == "METHOD":
            adjusted_score *= 3.0
        elif kind == "CLASS":
            adjusted_score *= 2.5
    elif has_test_annotation or has_scala_test:
        adjusted_score *= 2.0
    elif has_test_label:
        adjusted_score *= 1.8

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
    if kind in ignored_kinds:
        return adjusted_score
    node_id_lower = node_id.lower()
    code_lower = code.lower()
    if "test" in node_id_lower or "spec" in node_id_lower:
        if kind == "METHOD":
            adjusted_score *= 2.5
        elif kind == "CLASS":
            adjusted_score *= 2.0
    elif kind == "METHOD":
        if "override" in code_lower or "extends" in code_lower:
            adjusted_score *= 1.8
        elif len(code) > 200:
            adjusted_score *= 1.5
        else:
            adjusted_score *= 1.3
    elif kind in ["CLASS", "OBJECT", "TRAIT"]:
        adjusted_score *= 1.3
    elif kind == "CONSTRUCTOR":
        adjusted_score *= 1.2
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
    if kind in ignored_kinds:
        return adjusted_score
    code_lower = code.lower()
    if kind == "CLASS":
        adjusted_score *= 1.8
        if "case class" in code_lower:
            adjusted_score *= 1.1
    elif kind == "TRAIT":
        adjusted_score *= 1.7
    elif kind == "INTERFACE":
        adjusted_score *= 1.6
    elif kind == "OBJECT":
        adjusted_score *= 1.5
    elif kind == "CONSTRUCTOR":
        adjusted_score *= 1.4
    elif kind == "METHOD":
        adjusted_score *= 1.3
    elif kind == "TYPE":
        adjusted_score *= 1.2
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
    if kind in ignored_kinds:
        return adjusted_score
    code_lower = code.lower()
    if kind == "METHOD":
        adjusted_score *= 1.5
        if "override" in code_lower:
            adjusted_score *= 1.2
    elif kind == "CONSTRUCTOR":
        adjusted_score *= 1.3
    elif kind == "CLASS" and "abstract" in code_lower:
        adjusted_score *= 1.2
    elif kind == "TRAIT":
        adjusted_score *= 1.1
    elif kind == "OBJECT":
        adjusted_score *= 1.1

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
    if kind in ignored_kinds:
        return adjusted_score
    label = metadata.get("label", "")
    node_id_lower = node_id.lower()
    code_lower = code.lower()
    if "exception" in node_id_lower or "error" in node_id_lower:
        adjusted_score *= 2.0
    elif kind == "CLASS" and ("Exception" in label or "Error" in label):
        adjusted_score *= 1.8
    elif "throw " in code_lower or "catch" in code_lower:
        adjusted_score *= 1.5
    elif "recover" in code_lower:
        adjusted_score *= 1.4
    elif any(p in code for p in ["Try[", "Try {", "Either[", "Option["]):
        adjusted_score *= 1.2

    return adjusted_score


def general_factors(
    adjusted_score: float,
    node_id: str,
    metadata: Dict[str, Any],
    code: str,
    query: str,
    category: IntentCategory,
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
    label = metadata.get("label", "").lower()
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
    label_terms = set(label.split())
    node_id_terms = set(
        node_id.lower()
        .replace(".", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("#", " ")
        .replace("/", " ")
        .replace("!", " ")
        .replace("-", " ")
        .split()
    )
    term_overlap = len(query_terms.intersection(label_terms.union(node_id_terms)))
    if term_overlap > 0:
        adjusted_score *= 1.0 + 0.15 * term_overlap
    if kind in ignored_kinds:
        adjusted_score *= 0.6
    if category != IntentCategory.TESTING and (
        "test" in node_id.lower() or "spec" in node_id.lower()
    ):
        adjusted_score *= 0.8
    if confidence > 0.7:
        adjusted_score *= 1.0 + (confidence - 0.7) * 0.5

    return adjusted_score


def rerank_results(
    query: str, nodes: List[Tuple[float, Dict[str, Any]]], analysis: IntentAnalysis
) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Reranks retrieved nodes based on intent category and various heuristics.

    Args:
        query: Original user query text
        nodes: Retrieved items as (score, node_data)
        analysis: Analysis payload with category and confidence

    Returns:
        Reranked items sorted by adjusted score (descending)
    """
    category = getattr(analysis, "primary_intent", IntentCategory.GENERAL)
    confidence = getattr(analysis, "confidence", 0.5)

    try:
        intent_category = IntentCategory(category)
    except ValueError:
        intent_category = IntentCategory.GENERAL

    target_class_name = None
    if intent_category == IntentCategory.TESTING:
        target_class_name = extract_target_from_question(query)
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
            adjusted_score, node_id, metadata, code, query, intent_category, confidence
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