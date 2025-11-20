from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

max_usage_results = 10
max_public_interfaces = 2
max_other_usage = 1
max_test_usage = 4
code_hash_length = 200
debug_results_limit = 5
max_nodes_limit = 1000


def should_keep_method_definition(node_id: str, doc: str, target_class_name: str) -> bool:
    """
    Checks if a method definition should be kept as usage example.

    Args:
        node_id: Node identifier
        doc: Node code/document
        target_class_name: Target class name being searched

    Returns:
        True if method definition should be kept, False otherwise
    """
    is_test_method = "test" in node_id.lower()
    is_controller_endpoint = "controller" in node_id.lower() and any(
        annotation in doc
        for annotation in [
            "@GetMapping",
            "@PostMapping",
            "@PutMapping",
            "@DeleteMapping",
            "@RequestMapping",
        ]
    )

    if not is_test_method and not is_controller_endpoint:
        logger.debug(f"Skipping method definition: {node_id}")
        return False
    else:
        logger.debug(f"Keeping as usage example: {node_id}")
        return True


def detect_usage_pattern(
    doc: str, target_class_name: str
) -> Tuple[bool, float, Optional[str], Optional[str]]:
    """
    Detects usage patterns in the document.

    Args:
        doc: Document/code content
        target_class_name: Target class name to search for

    Returns:
        Tuple of (found_usage, score_boost, usage_type, pattern_key)
    """
    if f"{target_class_name}(" in doc:
        call_patterns = [
            f"= {target_class_name}(",
            f" {target_class_name}(",
            f"return {target_class_name}(",
            f".{target_class_name}(",
            f"({target_class_name}(",
        ]
        is_method_call = any(pattern in doc for pattern in call_patterns)
        method_definition_patterns = [
            f"public {target_class_name}(",
            f"private {target_class_name}(",
            f"protected {target_class_name}(",
        ]
        is_definition = any(pattern in doc for pattern in method_definition_patterns)
        if is_method_call and not is_definition:
            logger.debug(f"Method call: {target_class_name}() detected")
            return True, 0.4, "method_call", f"method_call_{target_class_name}"
    elif f".{target_class_name}(" in doc.lower():
        logger.debug(f"Service call: .{target_class_name}() detected")
        return True, 0.5, "service_call", f"service_call_{target_class_name}"

    return False, 0.0, None, None


def usage_score_boost(score: float, node_id: str, doc: str, usage_type: str) -> float:
    """
    Applies additional score boost for specific usage types.

    Args:
        score: Base score
        node_id: Node identifier
        doc: Document/code content
        usage_type: Type of usage detected

    Returns:
        Boosted score
    """
    if usage_type in ["method_call", "service_call"]:
        if "controller" in node_id.lower() and "test" not in node_id.lower():
            if any(
                mapping in doc
                for mapping in ["@GetMapping", "@PostMapping", "@PutMapping", "@DeleteMapping"]
            ):
                score += 0.3
                logger.debug(f"HTTP controller endpoint boost in {node_id}")
    return score


def filter_usage(
    usage_nodes: List[Tuple[float, str, str, Dict[str, Any]]], max_results: int
) -> List[Tuple[float, str, str, Dict[str, Any]]]:
    """
    Filters usage nodes to maintain a balanced mix of different types.

    Args:
        usage_nodes: List of usage nodes with scores
        max_results: Maximum number of results to return

    Returns:
        Filtered list maintaining type balance
    """
    filtered_usage = []
    controller_count = 0
    service_count = 0
    other_count = 0
    test_count = 0

    for score, node_id, doc, metadata in usage_nodes:
        node_lower = node_id.lower()

        if "test" in node_lower and test_count < max_test_usage:
            kind = metadata.get("kind", "")
            if kind == "CLASS":
                test_count += 1
            filtered_usage.append((score, node_id, doc, metadata))
        elif "controller" in node_lower and controller_count < max_public_interfaces:
            filtered_usage.append((score, node_id, doc, metadata))
            controller_count += 1
        elif "service" in node_lower and service_count < max_public_interfaces:
            filtered_usage.append((score, node_id, doc, metadata))
            service_count += 1
        elif other_count < max_other_usage:
            filtered_usage.append((score, node_id, doc, metadata))
            other_count += 1

        if len(filtered_usage) >= max_results:
            break

    return filtered_usage


def find_usage_nodes(
    collection: Any, target_class_name: str, max_results: int = max_usage_results
) -> List[Tuple[float, str, str, Dict[str, Any]]]:
    """
    Finds code nodes that use a given class/service.

    Args:
        collection: Chroma collection handle
        target_class_name: Class or service name to search usages for
        max_results: Maximum number of usage examples to return

    Returns:
        Sorted usage items as tuples: (score, node_id, code, metadata)
    """
    logger.debug(f"Searching for nodes that use: {target_class_name}")

    try:
        all_nodes = collection.get(limit=max_nodes_limit, include=["metadatas", "documents"])

        usage_nodes = []
        seen_patterns = set()

        for i in range(len(all_nodes["ids"])):
            node_id = all_nodes["ids"][i]
            metadata = all_nodes["metadatas"][i]
            doc = all_nodes["documents"][i]

            if not doc or doc.startswith("<"):
                continue

            # Check if it's method definition that should be kept
            if target_class_name.lower() in node_id.lower() and metadata.get("kind") == "METHOD":
                if not should_keep_method_definition(node_id, doc, target_class_name):
                    continue

            kind = metadata.get("kind", "")
            if kind in ["PARAMETER", "VARIABLE", "VALUE", "IMPORT"]:
                continue
            found_usage, score_boost, usage_type, pattern_key = detect_usage_pattern(
                doc, target_class_name
            )
            if not found_usage:
                continue
            if pattern_key:
                code_hash = hash(doc[:code_hash_length])
                unique_key = f"{pattern_key}_{code_hash}"

                if unique_key in seen_patterns:
                    logger.debug(f"Skipping duplicate: {node_id}")
                    continue
                seen_patterns.add(unique_key)
            if kind == "CONSTRUCTOR" and usage_type != "method_call":
                logger.debug(f"Skipping constructor without method call: {node_id}")
                continue
            score = 0.5 + score_boost
            score = usage_score_boost(score, node_id, doc, usage_type)
            usage_nodes.append((score, node_id, doc, metadata))

        usage_nodes.sort(key=lambda x: x[0], reverse=True)
        filtered_usage = filter_usage(usage_nodes, max_results)
        logger.debug(
            f"Found {len(filtered_usage)} usages after filtering (from {len(usage_nodes)} initial)"
        )
        if filtered_usage:
            logger.debug("Top usage nodes (after filtering):")
            for i, (score, node_id, doc, metadata) in enumerate(
                filtered_usage[:debug_results_limit]
            ):
                logger.debug(f"{i + 1}. {node_id} (score: {score:.3f})")
                logger.debug(f"Kind: {metadata.get('kind', 'UNKNOWN')}")
                logger.debug(f"Preview: {doc[:100]}...")

        return filtered_usage

    except Exception as e:
        logger.error(f"Error in find_usage_nodes: {e}")
        return []
