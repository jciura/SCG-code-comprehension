import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

from context.context_extraction import (
    extract_target_from_question,
    extract_usage_fragment,
)
from context.context_fallback import build_fallback_context
from context.context_filtering import (
    filter_definition_code,
    filter_exception_code,
    filter_implementation_code,
    filter_testing_code,
)
from context.context_priority import (
    get_max_sections_for_category,
    get_node_priority_score,
)
from src.core.intent_analyzer import IntentCategory, get_intent_analyzer

PHASE2_CHAR_THRESHOLD = 0.8
PHASE3_CHAR_THRESHOLD = 0.6
PHASE3_STOP_THRESHOLD = 0.9

MIN_CODE_LENGTH = 100
MAX_CODE_PREVIEW_SHORT = 1500
MAX_CODE_PREVIEW_LONG = 4000


def adjust_category(
        intent_category: IntentCategory,
        category: str,
        confidence: float,
        nodes: List[Tuple[float, Dict[str, Any]]],
) -> Tuple[IntentCategory, str, float]:
    """
    Adjust category based on first node's kind when confidence is low.

    Args:
        intent_category: Original detected intent
        category: Category string
        confidence: Detection confidence (0.0-1.0)
        nodes: Retrieved nodes

    Returns:
        Tuple of (adjusted_intent, adjusted_category, adjusted_confidence)
    """
    if intent_category == IntentCategory.GENERAL and confidence < 0.7 and nodes:
        first_node = nodes[0][1]
        node_kind = first_node.get("metadata", {}).get("kind", "")
        if node_kind in ["CLASS", "INTERFACE", "TRAIT", "OBJECT"]:
            return IntentCategory.DEFINITION, IntentCategory.DEFINITION.value, 0.8
        elif node_kind == "METHOD":
            return IntentCategory.IMPLEMENTATION, IntentCategory.IMPLEMENTATION.value, 0.7
    return intent_category, category, confidence


def filter_code_for_category(code: str, node_id: str, kind: str, category: str) -> Tuple[str, str]:
    """
    Filters code and creates header based on category.

    Returns:
        Tuple of (filtered_code, header)
    """
    if category == "exception":
        filtered_code = filter_exception_code(code, node_id)
        header = f"## {kind}: {node_id} (exception-handling)\n"
        return filtered_code, header
    elif category == "definition":
        filtered_code = filter_definition_code(code, node_id, kind)
        header = f"## {kind}: {node_id}\n"
        return filtered_code, header
    elif category == "testing":
        filtered_code = filter_testing_code(code, node_id)
        if not filtered_code:
            if len(code) > MAX_CODE_PREVIEW_SHORT:
                filtered_code = code[:MAX_CODE_PREVIEW_SHORT] + "..."
            else:
                filtered_code = code
        header = f"## {kind}: {node_id} (test)\n"
        return filtered_code, header
    elif category == "implementation":
        filtered_code = filter_implementation_code(code, node_id)
        if not filtered_code:
            if len(code) > MAX_CODE_PREVIEW_LONG:
                filtered_code = code[:MAX_CODE_PREVIEW_LONG] + "..."
            else:
                filtered_code = code
        header = f"## {kind}: {node_id} (implementation)\n"
        return filtered_code, header
    else:
        if category == "usage":
            max_length = MAX_CODE_PREVIEW_SHORT
        else:
            max_length = MAX_CODE_PREVIEW_LONG
        if len(code) > max_length:
            filtered_code = code[:max_length] + "..."
        else:
            filtered_code = code
        header = f"## {kind}: {node_id}\n"
        return filtered_code, header


def add_phase2_usage_nodes(
        remaining_nodes: List[Tuple[float, Dict[str, Any]]],
        target: Optional[str],
        add_node_func,
        category_nodes_limit: int,
) -> int:
    """Add usage-pattern nodes."""
    added = 0
    if not target:
        for score, node_data in remaining_nodes:
            if added >= category_nodes_limit:
                break
            node_id = node_data.get("node", "").lower()
            kind = node_data.get("metadata", {}).get("kind", "")
            if "test" in node_id or "spec" in node_id:
                if add_node_func(node_data, "usage-test", priority=95):
                    added += 1
            elif kind == "METHOD":
                if add_node_func(node_data, "usage-method", priority=80):
                    added += 1
        return added
    pattern = re.compile(rf"\b{re.escape(target)}\b", re.IGNORECASE)
    for score, node_data in remaining_nodes:
        if added >= category_nodes_limit:
            break
        node_id = node_data.get("node", "")
        code = node_data.get("code", "")
        node_name = node_id.split("/")[-1].split("#")[0].split(".")[-1].lower()
        if target.lower() == node_name:
            continue
        if not pattern.search(code):
            continue
        priority = 80 + min(int(score), 15)
        fragment = extract_usage_fragment(code, target, context_lines=5)
        if fragment:
            node_data_copy = node_data.copy()
            node_data_copy["code"] = fragment
            if add_node_func(node_data_copy, "usage-pattern", priority=priority):
                added += 1
        else:
            if add_node_func(node_data, "usage-pattern", priority=priority - 10):
                added += 1
    return added


def add_phase2_implementation_nodes(
        remaining_nodes: List[Tuple[float, Dict[str, Any]]],
        add_node_func,
        category_nodes_limit: int
) -> int:
    """Add implementation nodes."""
    added = 0
    for score, node_data in remaining_nodes:
        if added >= category_nodes_limit:
            break
        kind = node_data.get("metadata", {}).get("kind", "")
        code = node_data.get("code", "")
        is_implementation = (
                (kind == "METHOD" and len(code) > 100)
                or kind == "CONSTRUCTOR"
                or "override" in code.lower())
        if is_implementation:
            if add_node_func(node_data, "implementation", priority=75):
                added += 1
    return added


def add_phase2_testing_nodes(
        remaining_nodes: List[Tuple[float, Dict[str, Any]]],
        question: str,
        add_node_func,
        category_nodes_limit: int,
) -> int:
    """Add test-related nodes prioritized by target class match."""
    added = 0
    target_class = extract_target_from_question(question)
    for score, node_data in remaining_nodes:
        if added >= category_nodes_limit:
            break
        kind = node_data.get("metadata", {}).get("kind", "")
        code = node_data.get("code", "")
        node_id = node_data.get("node", "").lower()
        priority = 70
        is_test_node = (
                "test" in node_id
                or "spec" in node_id
                or "suite" in node_id)
        has_test_annotation = (
                "@test" in code.lower()
                or "@before" in code.lower()
                or "@after" in code.lower())
        has_test_keywords = any(kw in code.lower() for kw in [
            "assert", "should", "must", "expect", "verify"])
        if target_class and target_class.lower() in node_id:
            if kind == "METHOD" and is_test_node:
                priority = 95
            elif kind == "CLASS" and is_test_node:
                priority = 90
            elif kind == "METHOD" and has_test_keywords:
                priority = 88
        elif kind == "METHOD" and (is_test_node or has_test_annotation):
            priority = 85
        elif kind == "CLASS" and is_test_node:
            priority = 82
        elif has_test_keywords:
            priority = 75
        if priority > 70 or is_test_node or has_test_annotation:
            if add_node_func(node_data, "test-method", priority=priority):
                added += 1
    return added


def add_phase2_definition_nodes(
        remaining_nodes: List[Tuple[float, Dict[str, Any]]],
        add_node_func,
        category_nodes_limit: int
) -> int:
    """Add definition nodes."""
    added = 0
    for score, node_data in remaining_nodes:
        if added >= category_nodes_limit:
            break
        kind = node_data.get("metadata", {}).get("kind", "")
        node_id = node_data.get("node", "").lower()
        if kind in ["CLASS", "INTERFACE", "TRAIT", "OBJECT"]:
            if add_node_func(node_data, "definition-class", priority=90):
                added += 1
        elif kind == "CONSTRUCTOR":
            if add_node_func(node_data, "definition-constructor", priority=85):
                added += 1
        elif kind == "METHOD" and ("get" not in node_id and "set" not in node_id):
            if add_node_func(node_data, "definition-method", priority=75):
                added += 1
    return added


def add_phase2_exception_nodes(
        remaining_nodes: List[Tuple[float, Dict[str, Any]]],
        add_node_func,
        category_nodes_limit: int
) -> int:
    """Add exception-handling nodes."""
    added = 0
    for score, node_data in remaining_nodes:
        if added >= category_nodes_limit:
            break
        code = node_data.get("code", "")
        node_id = node_data.get("node", "").lower()
        code_lower = code.lower()
        if "exception" in node_id or "error" in node_id:
            if add_node_func(node_data, "exception-class", priority=95):
                added += 1
        elif "throw" in code_lower or "catch" in code_lower:
            if add_node_func(node_data, "exception-handler", priority=85):
                added += 1
        elif "try" in code_lower or "recover" in code_lower:
            if add_node_func(node_data, "exception-try", priority=75):
                added += 1
    return added


def add_phase2_general_nodes(
        remaining_nodes: List[Tuple[float, Dict[str, Any]]],
        add_node_func,
        category_nodes_limit: int
) -> int:
    """Add general-purpose nodes."""
    added = 0
    for score, node_data in remaining_nodes:
        if added >= category_nodes_limit:
            break
        kind = node_data.get("metadata", {}).get("kind", "")
        code = node_data.get("code", "")
        if kind in ["CLASS", "TRAIT", "OBJECT"]:
            if add_node_func(node_data, "general-class", priority=85):
                added += 1
        elif kind == "METHOD" and len(code) > MAX_CODE_PREVIEW_SHORT:
            if add_node_func(node_data, "general-method", priority=75):
                added += 1
        elif kind == "INTERFACE":
            if add_node_func(node_data, "general-interface", priority=80):
                added += 1
    return added


def phase2(
        category: str,
        remaining_nodes: List[Tuple[float, Dict[str, Any]]],
        question: str,
        target: Optional[str],
        add_node_func,
        category_nodes_limit: int,
) -> int:
    """
    Apply category-specific heuristics to add secondary context nodes.
    """
    if category == "usage":
        return add_phase2_usage_nodes(
            remaining_nodes, target, add_node_func, category_nodes_limit
        )
    elif category == "implementation":
        return add_phase2_implementation_nodes(remaining_nodes, add_node_func, category_nodes_limit)
    elif category == "testing":
        return add_phase2_testing_nodes(
            remaining_nodes, question, add_node_func, category_nodes_limit
        )
    elif category == "definition":
        return add_phase2_definition_nodes(remaining_nodes, add_node_func, category_nodes_limit)
    elif category == "exception":
        return add_phase2_exception_nodes(remaining_nodes, add_node_func, category_nodes_limit)
    elif category == "general":
        return add_phase2_general_nodes(remaining_nodes, add_node_func, category_nodes_limit)
    return 0


def build_context(
        nodes: List[Tuple[float, Dict[str, Any]]],
        category: str,
        confidence: float,
        top_nodes_limit: int = 1,
        question: str = "",
        target: Optional[str] = None,
) -> str:
    """
    Builds an intent-aware, token-bounded context string from retrieved nodes.

    Args:
        nodes: Retrieved nodes with similarity scores and payloads
        category: Detected/assumed question category
        confidence: Confidence for the category (0.0–1.0)
        top_nodes_limit: Number of top_nodes extracted from question
        question: Original user question (used to infer targets)
        target: Explicit target entity (method/class) to search for usages

    Returns:
        Final context string composed of titled sections
    """
    logger.debug(f"\nbuild_context start - Category: {category}, Confidence: {confidence:.2f}")
    logger.debug(f"Target: {target}, Number of nodes: {len(nodes)}")

    if not nodes:
        logger.debug("No nodes provided, using fallback")
        return build_fallback_context()
    try:
        intent_category = IntentCategory(category)
    except ValueError:
        intent_category = IntentCategory.GENERAL
    intent_category, category, confidence = adjust_category(
        intent_category, category, confidence, nodes
    )
    analyzer = get_intent_analyzer()
    limits = analyzer.get_context_limits(intent_category)
    max_context_chars = limits["max_chars"]
    base_nodes_limit = limits["base_nodes"]
    category_nodes_limit = limits["category_nodes"]
    fill_nodes_limit = limits["fill_nodes"]
    if confidence >= 0.8:
        base_nodes_limit = min(base_nodes_limit + 1, 3)
    elif confidence < 0.5:
        base_nodes_limit = max(base_nodes_limit - 1, 1)
    if not target and category == "usage":
        target = extract_target_from_question(question)
    context_sections: List[Tuple[int, str]] = []
    current_chars = 0
    seen_codes: Set[str] = set()
    used_nodes: Set[str] = set()
    section_counts = Counter()

    def add_node_section(node_data: Dict[str, Any], section_label: str, priority: int = 0) -> bool:
        nonlocal current_chars
        code = node_data.get("code", "")
        node_id = node_data.get("node", "")
        metadata = node_data.get("metadata", {})
        kind = metadata.get("kind", "CODE")
        if (
                not code
                or len(code) < MIN_CODE_LENGTH
                or code in seen_codes
                or node_id in used_nodes
                or code.startswith("<")
        ):
            return False
        max_sections = get_max_sections_for_category(category)
        if section_counts[section_label] >= max_sections:
            return False
        code_preview, header = filter_code_for_category(code, node_id, kind, category)
        if not code_preview:
            logger.debug("No Code preview")
            return False

        full_section = header + code_preview + "\n"
        section_chars = len(full_section)
        if current_chars + section_chars <= max_context_chars - 50:
            logger.debug(
                f"Current chars: {current_chars}, "
                f"Section chars: {section_chars},"
                f" Max chars: {max_context_chars}"
            )
            context_sections.append((priority, full_section))
            seen_codes.add(code)
            used_nodes.add(node_id)
            current_chars += section_chars
            section_counts[section_label] += 1
            logger.debug(f"Added {node_id} ({kind}), chars: {section_chars}")
            return True

        logger.debug(f"Not added {node_id} ({kind}), chars: {section_chars}")
        return False

    scored_nodes = []
    for score, node_data in nodes:
        priority_score = get_node_priority_score(node_data, category, target)
        combined_score = score + priority_score * 0.1
        scored_nodes.append((combined_score, node_data))
    base_nodes = scored_nodes[:top_nodes_limit]
    rest_nodes = scored_nodes[top_nodes_limit:]
    rest_nodes.sort(key=lambda x: x[0], reverse=True)
    scored_nodes = base_nodes + rest_nodes

    for score, node_data in scored_nodes:
        logger.info(node_data.get("node", ""))

    added_base = 0
    for score, node_data in scored_nodes[: base_nodes_limit * 2]:
        if added_base >= base_nodes_limit:
            break
        if add_node_section(node_data, "top-match", priority=100):
            added_base += 1
    logger.debug(f"Phase 1: Added {added_base} base nodes")

    if category_nodes_limit > 0 and current_chars < max_context_chars * PHASE2_CHAR_THRESHOLD:
        remaining_nodes = [(s, n) for s, n in scored_nodes if n.get("node", "") not in used_nodes]
        added_category = phase2(
            category,
            remaining_nodes,
            question,
            target,
            add_node_section,
            category_nodes_limit,
        )
        logger.debug(f"Phase 2: Added {added_category} category-specific nodes")

    if fill_nodes_limit > 0 and current_chars < max_context_chars * PHASE3_CHAR_THRESHOLD:
        final_remaining = [(s, n) for s, n in scored_nodes if n.get("node", "") not in used_nodes]
        added_fill = 0
        for score, node_data in final_remaining[:fill_nodes_limit]:
            if current_chars >= max_context_chars * PHASE3_STOP_THRESHOLD:
                break
            if add_node_section(node_data, "additional", priority=50):
                added_fill += 1
        logger.debug(f"Phase 3: Added {added_fill} fill nodes")

    if not context_sections:
        logger.debug("No context sections created, using fallback")
        return build_fallback_context()

    context_sections.sort(key=lambda x: x[0], reverse=True)
    final_context = "\n\n".join(section[1] for section in context_sections)

    logger.debug(f"Final: {len(context_sections)} sections, {current_chars} chars")

    return final_context