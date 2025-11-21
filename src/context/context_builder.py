from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

from context.context_extraction import (
    extract_method_with_context,
    extract_target_from_question,
    extract_usage_fragment,
)
from context.context_fallback import build_fallback_context
from context.context_filtering import (
    filter_definition_code,
    filter_exception_code,
)
from context.context_priority import (
    get_max_sections_for_category,
    get_node_priority_score,
)
from context.context_tokens import estimate_tokens
from src.core.intent_analyzer import IntentCategory, get_intent_analyzer


def build_context(
    nodes: List[Tuple[float, Dict[str, Any]]],
    category: str,
    confidence: float,
    question: str = "",
    target_method: Optional[str] = None,
) -> str:
    """
    Builds an intent-aware, token-bounded context string from retrieved nodes.

    Args:
        nodes: Retrieved nodes with similarity scores and payloads
        category: Detected/assumed question category
        confidence: Confidence for the category (0.0–1.0)
        question: Original user question (used to infer targets)
        target_method: Explicit target method to search for usages

    Returns:
        Final context string composed of titled sections
    """
    logger.debug("\n build context start ")
    logger.debug(f"Category: {category}, Confidence: {confidence:.2f}")
    logger.debug(f"Target method: {target_method}")
    logger.debug(f"Number of nodes: {len(nodes)}")

    if not nodes:
        logger.debug("No nodes provided, using fallback")
        return build_fallback_context()

    for i, (score, node_data) in enumerate(nodes[:5]):
        logger.debug(
            f"Node {i}: {node_data.get('node', 'Unknown')} "
            f"({node_data.get('metadata', {}).get('kind', 'Unknown')}) "
            f"score={score:.3f}"
        )

    try:
        intent_category = IntentCategory(category)
    except ValueError:
        intent_category = IntentCategory.GENERAL

    if category == "definition" and nodes:
        best_node = nodes[0][1]
        kind = best_node.get("metadata", {}).get("kind", "")
        if kind == "CLASS":
            code = best_node.get("code", "")
            node_id = best_node.get("node", "")
            if code:
                filtered_code = filter_definition_code(code, node_id, kind)
                if filtered_code:
                    result = f"## {kind}: {node_id}\n{filtered_code}"
                    logger.debug(f"return for CLASS definition: {len(result)} chars")
                    return result

    if category in ["medium", "general"] and confidence < 0.7 and nodes:
        first_node = nodes[0][1]
        node_kind = first_node.get("metadata", {}).get("kind", "")
        if node_kind in ["CLASS", "INTERFACE"]:
            category = "definition"
            confidence = 0.8
        elif node_kind == "METHOD":
            category = "implementation"
            confidence = 0.7

    analyzer = get_intent_analyzer()
    limits = analyzer.get_context_limits(intent_category)

    max_context_tokens = limits["max_tokens"]
    base_nodes_limit = limits["base_nodes"]
    category_nodes_limit = limits["category_nodes"]
    fill_nodes_limit = limits["fill_nodes"]
    if confidence >= 0.8:
        base_nodes_limit = min(base_nodes_limit + 1, 3)
    elif confidence < 0.5:
        base_nodes_limit = max(base_nodes_limit - 1, 1)
    if not target_method and category == "usage":
        target_method = extract_target_from_question(question)
    context_sections = []
    current_tokens = 0
    seen_codes: Set[str] = set()
    used_nodes: Set[str] = set()
    section_counts = Counter()

    def add_node_section(node_data: Dict[str, Any], section_label: str, priority: int = 0) -> bool:
        nonlocal current_tokens
        code = node_data.get("code", "")
        node_id = node_data.get("node", "")
        metadata = node_data.get("metadata", {})
        kind = metadata.get("kind", "CODE")

        if (
            not code
            or len(code) < 20
            or code in seen_codes
            or node_id in used_nodes
            or code.startswith("<")
        ):
            return False

        max_sections = get_max_sections_for_category(category)
        if section_counts[section_label] >= max_sections:
            return False

        if kind == "METHOD" and category in ["definition", "usage", "implementation"]:
            context_code = extract_method_with_context(metadata, context_lines=5)
            if context_code and not context_code.startswith("<"):
                code_preview = context_code
                header = f"## {kind}: {node_id} (with context)\n"
            else:
                max_length = 150 if category in ["usage", "testing"] else 200
                code_preview = code[:max_length] + "..." if len(code) > max_length else code
                header = f"## {kind}: {node_id}\n"
        elif category == "exception":
            filtered_code = filter_exception_code(code)
            if not filtered_code:
                return False
            code_preview = filtered_code
            header = f"## {kind}: {node_id} (exception-handling)\n"
        elif category == "definition":
            filtered_code = filter_definition_code(code, node_id, kind)
            if not filtered_code:
                return False
            code_preview = filtered_code
            header = f"## {kind}: {node_id}\n"
        else:
            max_length = 150 if category in ["usage", "testing"] else 200
            code_preview = code[:max_length] + "..." if len(code) > max_length else code
            header = f"## {kind}: {node_id}\n"

        full_section = header + code_preview + "\n"
        section_tokens = estimate_tokens(full_section, is_code=True)

        logger.debug(f"add_node_section: Trying to add {node_id}")
        logger.debug(f"Kind: {kind}, Label: {section_label}, Priority: {priority}")
        logger.debug(
            f"Tokens: {section_tokens}, Current: {current_tokens}, Limit: {max_context_tokens}"
        )

        if current_tokens + section_tokens <= max_context_tokens - 10:
            context_sections.append((priority, full_section))
            seen_codes.add(code)
            used_nodes.add(node_id)
            current_tokens += section_tokens
            section_counts[section_label] += 1
            logger.debug("added successfully")
            return True

        logger.debug("rejected - exceed token limit")
        return False

    scored_nodes = []
    for score, node_data in nodes:
        priority_score = get_node_priority_score(node_data, category)
        combined_score = score + priority_score * 0.1
        scored_nodes.append((combined_score, node_data))
    scored_nodes.sort(key=lambda x: x[0], reverse=True)

    added_base = 0
    for score, node_data in scored_nodes[: base_nodes_limit * 2]:
        if added_base >= base_nodes_limit:
            break
        if add_node_section(node_data, "top-match", priority=100):
            added_base += 1
    logger.debug(f"Phase 1: Added {added_base} base nodes")

    if category_nodes_limit > 0 and current_tokens < max_context_tokens * 0.8:
        remaining_nodes = [(s, n) for s, n in scored_nodes if n.get("node", "") not in used_nodes]
        added_category = 0

        if category == "usage":
            for score, node_data in remaining_nodes:
                if added_category >= category_nodes_limit:
                    break
                node_id = node_data.get("node", "").lower()
                code = node_data.get("code", "")
                metadata = node_data.get("metadata", {})
                kind = metadata.get("kind", "")

                if "test" in node_id and kind in ["CLASS", "METHOD"]:
                    if add_node_section(node_data, "usage-test", priority=95):
                        added_category += 1
                    continue

                if kind in ["CLASS", "CONSTRUCTOR"] and target_method:
                    fragment = extract_usage_fragment(code, target_method, context_lines=5)
                    if fragment:
                        node_data_copy = node_data.copy()
                        node_data_copy["code"] = fragment
                        if add_node_section(node_data_copy, "usage-pattern", priority=90):
                            added_category += 1
                        continue

                if (
                    ("controller" in node_id and "test" not in node_id)
                    or ("service" in node_id and "test" not in node_id)
                    or kind == "METHOD"
                ):
                    if add_node_section(node_data, "usage-pattern", priority=80):
                        added_category += 1

        elif category == "implementation":
            for score, node_data in remaining_nodes:
                if added_category >= category_nodes_limit:
                    break
                kind = node_data.get("metadata", {}).get("kind", "")
                code = node_data.get("code", "")
                if (
                    (kind == "METHOD" and len(code) > 100)
                    or kind == "CONSTRUCTOR"
                    or "algorithm" in code.lower()
                ):
                    if add_node_section(node_data, "implementation", priority=75):
                        added_category += 1

        elif category == "testing":
            target_class = extract_target_from_question(question)
            for score, node_data in remaining_nodes:
                if added_category >= category_nodes_limit:
                    break
                kind = node_data.get("metadata", {}).get("kind", "")
                code = node_data.get("code", "")
                node_id = node_data.get("node", "").lower()
                priority = 70

                if target_class and target_class in node_id:
                    if kind == "METHOD" and "test" in node_id:
                        priority = 95
                    elif kind == "CLASS" and "test" in node_id:
                        priority = 90
                    elif kind == "METHOD" and any(
                        w in node_data.get("metadata", {}).get("label", "").lower()
                        for w in ["should", "test"]
                    ):
                        priority = 88
                elif kind == "METHOD" and (
                    "test" in node_id or "@test" in code.lower() or "should" in node_id
                ):
                    priority = 85
                elif kind == "CLASS" and "test" in node_id:
                    priority = 82

                if priority > 70 or (kind in ["METHOD", "CLASS"] and "test" in node_id):
                    if add_node_section(node_data, "test-method", priority=priority):
                        added_category += 1

        logger.debug(f"Phase 2: Added {added_category} category-specific nodes")

    if fill_nodes_limit > 0 and current_tokens < max_context_tokens * 0.6:
        final_remaining = [(s, n) for s, n in scored_nodes if n.get("node", "") not in used_nodes]
        added_fill = 0
        for score, node_data in final_remaining[:fill_nodes_limit]:
            if current_tokens >= max_context_tokens * 0.9 or added_fill >= fill_nodes_limit:
                break
            if add_node_section(node_data, "additional", priority=50):
                added_fill += 1
        logger.debug(f"Phase 3: Added {added_fill} fill nodes")

    if not context_sections:
        logger.debug("No context sections created, using fallback")
        return build_fallback_context()

    context_sections.sort(key=lambda x: x[0], reverse=True)
    final_context = "\n\n".join(section[1] for section in context_sections)

    logger.debug("\n final context")
    logger.debug(f"Number of sections: {len(context_sections)}")
    logger.debug(f"Final context length: {len(final_context)} chars")
    logger.debug(f"Number of ## headers in final context: {final_context.count('##')}")

    if final_context.count("##") == 0:
        logger.debug("No ## headers found, regenerating with headers")
        fixed_sections = []
        for priority, section in context_sections:
            if not section.strip().startswith("##"):
                fixed_sections.append(f"## CODE SECTION\n{section}")
            else:
                fixed_sections.append(section)
        final_context = "\n\n".join(fixed_sections)

    logger.debug(
        f"Tokens: {current_tokens}, Sections: {len(context_sections)}, Chars: {len(final_context)}"
    )
    logger.debug(f"Headers count: {final_context.count('##')}")

    return final_context
