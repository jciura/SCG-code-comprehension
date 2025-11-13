from typing import List, Tuple, Dict, Any, Set, Optional
import time
import re
import os
from collections import Counter
from src.core.intent_analyzer import get_intent_analyzer, IntentCategory
from loguru import logger

max_context_tokens = 2000
code_chars_per_token = 3.5
text_chars_per_token = 4.2
min_code_length = 30
fallback_cache_duration = 300

_context_cache = {
    'fallback': None,
    'timestamp': 0,
    'stats': {'hits': 0, 'misses': 0}
}


def extract_method_with_context(metadata: Dict[str, Any], context_lines: int = 5) -> str:
    """
        Extracts a method's source code along with surrounding context lines.

        Opens the file specified in metadata, locates the method start by line number,
        and captures the full code block including braces and nearby lines for context.

        Args:
            metadata (Dict[str, Any]): Node metadata containing 'uri' and 'location'.
            context_lines (int, optional): Number of context lines to include before and after.
                Defaults to 5.

        Returns:
            str: Method code with line numbers and context, or an error message if extraction fails.
    """
    try:
        uri = metadata.get('uri', '')
        location = metadata.get('location', '')
        if not uri or not location:
            return ""
        if uri.startswith('file://'):
            uri = uri[7:]
        if not os.path.exists(uri):
            return ""
        start, _ = location.split(';')
        start_line, _ = map(int, start.split(':'))
        with open(uri, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if start_line > len(lines):
            return f"<Invalid start line: {start_line}>"
        start_idx = max(0, start_line - 1 - context_lines)
        open_braces = 0
        end_idx = start_line - 1
        method_started = False
        for i in range(start_line - 1, len(lines)):
            line = lines[i]
            open_braces += line.count('{') - line.count('}')
            if not method_started and '{' in line:
                method_started = True
            if method_started and open_braces == 0:
                end_idx = i
                break
        else:
            end_idx = min(len(lines) - 1, start_line + 20)

        end_idx = min(len(lines) - 1, end_idx + context_lines)
        context_lines_list = lines[start_idx:end_idx + 1]
        numbered_lines = []
        for i, line in enumerate(context_lines_list):
            line_num = start_idx + i + 1
            prefix = ">>>" if start_line <= line_num <= start_line + 2 else "   "
            numbered_lines.append(f"{prefix} {line_num:3d}: {line.rstrip()}")

        return '\n'.join(numbered_lines)
    except Exception as e:
        return f"<Could not extract method with context: {e}>"


def filter_definition_code(code: str, node_id: str, kind: str) -> str:
    """
        Filters and summarizes definition-related code sections.

        Keeps only key structural elements such as class/interface declarations,
        field signatures, constructors, and method headers.

        Args:
            code (str): Full code text to process.
            node_id (str): Identifier of the graph node (used to infer class name).
            kind (str): Node type, e.g. "CLASS" or "INTERFACE".

        Returns:
            str: Filtered code snippet containing main definitions.
    """
    if not code or kind not in ["CLASS", "INTERFACE"]:
        return code[:400] if code else ""

    definition_lines = []
    lines = code.split('\n')
    for line in lines:
        line_clean = line.strip()
        if not line_clean or line_clean.startswith('//') or line_clean.startswith('*'):
            continue
        if (line_clean.startswith('@') or
                line_clean.startswith('public class') or
                line_clean.startswith('public interface') or
                line_clean.startswith('private class') or
                line_clean.startswith('protected class')):
            definition_lines.append(line_clean)
            continue
        if ((line_clean.startswith('private ') or
             line_clean.startswith('protected ') or
             line_clean.startswith('public ')) and
                '(' not in line_clean and
                '=' not in line_clean):
            definition_lines.append(line_clean)
            continue
        class_name = node_id.split('.')[-1] if '.' in node_id else node_id
        if (('public ' + class_name + '(') in line_clean or
                ('private ' + class_name + '(') in line_clean or
                ('protected ' + class_name + '(') in line_clean):
            definition_lines.append(line_clean)
            continue
        if (any(modifier in line_clean for modifier in ['public ', 'private ', 'protected ']) and
                '(' in line_clean and ')' in line_clean and
                not line_clean.startswith('@')):
            method_signature = line_clean[:-1].strip() + ';' if line_clean.endswith('{') else line_clean + ';'
            definition_lines.append(method_signature)
            continue
        if line_clean == '}' and len(definition_lines) > 5:
            definition_lines.append(line_clean)
            break

    return '\n'.join(definition_lines[:15])


def filter_exception_code(code: str) -> str:
    """
        Extracts exception-related lines from code.

        Captures lines with `throw new`, `orElseThrow`, or words ending with
        'Exception' or 'Error', including relevant import statements.

        Args:
            code (str): Code text to analyze.

        Returns:
            str: Extracted exception-related lines joined as a string.
    """
    if not code:
        return ""

    exception_lines = []
    lines = code.split('\n')
    for line in lines:
        line_clean = line.strip()
        if 'throw new' in line_clean or 'orElseThrow(' in line_clean:
            exception_lines.append(line_clean)
            continue
        words = re.split(r'[().,;{}\s]+', line_clean)
        has_exception = any(word.endswith(('Exception', 'Error')) and len(word) > 5 for word in words)
        if has_exception:
            exception_lines.append(line_clean)
            continue
        if line_clean.startswith('import') and ('Exception' in line_clean or 'Error' in line_clean):
            exception_lines.append(line_clean)
    return '\n'.join(exception_lines[:40])


def extract_usage_fragment(code: str, target_method: str, context_lines: int = 5) -> Optional[str]:
    """
        Extracts a fragment of code showing usage of a target method.

        Searches for occurrences of `target_method(` and returns nearby lines
        for contextual understanding.

        Args:
            code (str): Source code text.
            target_method (str): Method name to locate.
            context_lines (int, optional): Number of lines before and after to include.
                Defaults to 5.

        Returns:
            Optional[str]: Code fragment containing the method call, or None if not found.
    """
    if not target_method or f'{target_method}(' not in code:
        return None
    code_lines = code.split('\n')
    for i, line in enumerate(code_lines):
        if f'{target_method}(' in line:
            start = max(0, i - context_lines)
            end = min(len(code_lines), i + context_lines + 1)
            return '\n'.join(code_lines[start:end])
    return None


def estimate_tokens(text: str, is_code: bool = True) -> int:
    """
        Estimates token count for a text snippet.

        Uses a heuristic based on average character-per-token ratios for code or plain text.

        Args:
            text (str): Input text to estimate.
            is_code (bool, optional): Whether the input is source code.
                Defaults to True.

        Returns:
            int: Estimated token count (minimum 1).
    """
    if not text:
        return 0
    chars_per_token = code_chars_per_token if is_code else text_chars_per_token
    non_whitespace_chars = len(text.replace(' ', '').replace('\n', '').replace('\t', ''))
    whitespace_ratio = 1 - (non_whitespace_chars / len(text)) if len(text) > 0 else 0
    adjusted_ratio = chars_per_token * (1 + whitespace_ratio * 0.3)
    return max(1, int(len(text) / adjusted_ratio))


def get_node_priority_score(node_data: Dict[str, Any], category: str) -> float:
    """
        Computes priority score for a graph node.

        Combines structural importance metrics based
        on the node type, category, and code content (e.g., testing, usage, exception).

        Args:
            node_data (Dict[str, Any]): Node data containing 'metadata', 'code', and 'node'.
            category (str): Detected question category (e.g., "usage", "definition").

        Returns:
            float: Calculated node priority score.
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
    elif len(code) < min_code_length:
        score -= 3
    if kind in ["PARAMETER", "VARIABLE", "IMPORT"] and category not in ["definition"]:
        score -= 5
    return score


def get_max_sections_for_category(category: str) -> int:
    """
        Returns the maximum number of code sections to include per category.

        Args:
            category (str): Question category (e.g., "testing", "usage").

        Returns:
            int: Maximum number of sections allowed for the given category.
    """
    return {
        "testing": 8,
        "usage": 5,
        "implementation": 3,
        "exception": 3,
        "definition": 2,
        "general": 4,
    }.get(category, 2)


def extract_target_from_question(question: str) -> Optional[str]:
    """
        Extracts a probable target entity (method/class) name from a question.

        Uses regex patterns to detect references like "method X", "class Y",
        "tests for Z", or "find usage of FooService".

        Args:
            question (str): Natural language question text.

        Returns:
            Optional[str]: Extracted target name, or None if no match found.
    """
    if not question:
        return None

    question_lower = question.lower()
    patterns = [
        r'method\s+(\w+)',
        r'(\w+)\s+method',
        r'class\s+(\w+)',
        r'(\w+)\s+class',
        r'for\s+(\w+)\s+class',
        r'where.*\s+(\w+)\s+used',
        r'tests?\s+for\s+(\w+)',
        r'(\w+controller)',
        r'(\w+service)',
        r'(\w+repository)',
    ]

    for pattern in patterns:
        match = re.search(pattern, question_lower)
        if match:
            target = match.group(1)
            return target
    return None


def build_context(nodes: List[Tuple[float, Dict[str, Any]]], category: str, confidence: float, question: str = "", target_method: str = None) -> str:
    """
        Builds an intent-aware, token-bounded context string from retrieved nodes.

        Selects and formats code sections based on detected category (e.g., usage,
        definition, implementation, testing, exception), confidence, and optional
        target method. Applies token limits from the intent analyzer, prioritizes
        high-importance nodes, adds neighbors when useful, and falls back when empty.

        Args:
            nodes (List[Tuple[float, Dict[str, Any]]]): Retrieved nodes with similarity
                scores and payloads containing `node`, `metadata`, and `code`.
            category (str): Detected/assumed question category (e.g., "usage", "general").
            confidence (float): Confidence for the category (0.0–1.0).
            question (str, optional): Original user question (used to infer targets).
                Defaults to "".
            target_method (str, optional): Explicit target method to search for usages.
                If None and category is "usage", it may be inferred from the question.

        Returns:
            str: Final context string composed of titled sections, or a fallback context
            if no suitable sections can be assembled.

        Notes:
            - Respects token budget from `get_intent_analyzer().get_context_limits(...)`.
            - Uses helpers like `extract_method_with_context`, `filter_definition_code`,
              `filter_exception_code`, `estimate_tokens`, and `get_node_priority_score`.
    """
    logger.debug(f"\n build context start ")
    logger.debug(f"Category: {category}, Confidence: {confidence:.2f}")
    logger.debug(f"Target method: {target_method}")
    logger.debug(f"Number of nodes: {len(nodes)}")

    if not nodes:
        logger.debug("No nodes provided, using fallback")
        return build_fallback_context()
    for i, (score, node_data) in enumerate(nodes[:5]):
        logger.debug(f"Node {i}: {node_data.get('node', 'Unknown')} "
                     f"({node_data.get('metadata', {}).get('kind', 'Unknown')}) "
                     f"score={score:.3f}")
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
        if (not code or len(code) < 20 or code in seen_codes or
                node_id in used_nodes or code.startswith('<')):
            return False

        max_sections = get_max_sections_for_category(category)
        if section_counts[section_label] >= max_sections:
            return False

        if kind == "METHOD" and category in ["definition", "usage", "implementation"]:
            context_code = extract_method_with_context(metadata, context_lines=5)
            if context_code and not context_code.startswith('<'):
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
        logger.debug(f"Has ## header: {full_section.startswith('##')}")
        logger.debug(f"Section preview: {full_section[:80]}...")
        logger.debug(f"Tokens: {section_tokens}, Current: {current_tokens}, Limit: {max_context_tokens}")

        if current_tokens + section_tokens <= max_context_tokens - 10:
            context_sections.append((priority, full_section))
            seen_codes.add(code)
            used_nodes.add(node_id)
            current_tokens += section_tokens
            section_counts[section_label] += 1
            logger.debug(f"added successfully")
            return True
        else:
            logger.debug(f"rejected - exceed token limit")
            return False

    scored_nodes = []
    for score, node_data in nodes:
        priority_score = get_node_priority_score(node_data, category)
        combined_score = score + priority_score * 0.1
        scored_nodes.append((combined_score, node_data))

    scored_nodes.sort(key=lambda x: x[0], reverse=True)
    added_base = 0
    for score, node_data in scored_nodes[:base_nodes_limit * 2]:
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
                        node_data_copy['code'] = fragment
                        if add_node_section(node_data_copy, "usage-pattern", priority=90):
                            added_category += 1
                        continue

                if (("controller" in node_id and "test" not in node_id) or
                        ("service" in node_id and "test" not in node_id) or
                        kind == "METHOD"):
                    if add_node_section(node_data, "usage-pattern", priority=80):
                        added_category += 1

        elif category == "implementation":
            for score, node_data in remaining_nodes:
                if added_category >= category_nodes_limit:
                    break

                kind = node_data.get("metadata", {}).get("kind", "")
                code = node_data.get("code", "")

                if ((kind == "METHOD" and len(code) > 100) or
                        kind == "CONSTRUCTOR" or
                        "algorithm" in code.lower()):
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
                    elif kind == "METHOD" and any(w in node_data.get("metadata", {}).get("label", "").lower() for w in ["should", "test"]):
                        priority = 88
                elif (kind == "METHOD" and ("test" in node_id or "@test" in code.lower() or "should" in node_id)):
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

    logger.debug(f"\n final context")
    logger.debug(f"Number of sections: {len(context_sections)}")
    logger.debug(f"Final context length: {len(final_context)} chars")
    logger.debug(f"Number of ## headers in final context: {final_context.count('##')}")
    logger.debug(f"First 500 chars of final context:")
    logger.debug(final_context[:500])
    logger.debug("\nSection priorities:")
    for i, (priority, section) in enumerate(context_sections[:5]):
        first_line = section.split('\n')[0] if '\n' in section else section[:80]
        logger.debug(f"  {i}. Priority={priority}: {first_line}")

    if final_context.count('##') == 0:
        logger.debug("No ## headers found, regenerating with headers")

        fixed_sections = []

        for priority, section in context_sections:
            if not section.strip().startswith('##'):
                fixed_sections.append(f"## CODE SECTION\n{section}")
            else:
                fixed_sections.append(section)

        final_context = "\n\n".join(fixed_sections)

    logger.debug(f"\n final context")
    logger.debug(f"Tokens: {current_tokens}, Sections: {len(context_sections)}, Chars: {len(final_context)}")
    logger.debug(f"Headers count: {final_context.count('##')}")

    return final_context


def build_fallback_context(collection=None) -> str:
    """
        Builds (and caches) a fallback context from high-importance nodes.

        When primary selection yields no sections, this function pulls top documents
        by importance from the Chroma collection and composes a compact context.
        Employs a simple in-memory cache to avoid repeated work.

        Args:
            collection: Optional Chroma collection handle. If None, it will try to
                obtain `scg_embeddings` from `src.graph.retriver.chroma_client`.

        Returns:
            str: Fallback context string, or an error marker if unavailable.

        Notes:
            - Uses a time-bounded cache (`_context_cache`) with hit/miss statistics.
            - Token budget for fallback is set to roughly half of `max_context_tokens`.
    """
    global _context_cache
    current_time = time.time()
    if (_context_cache['fallback'] is not None and
            current_time - _context_cache['timestamp'] < fallback_cache_duration):
        _context_cache['stats']['hits'] += 1
        logger.debug("Using cached fallback context")
        return _context_cache['fallback']

    _context_cache['stats']['misses'] += 1
    if collection is None:
        try:
            from src.graph.retriver import chroma_client
            collection = chroma_client.get_collection(name="scg_embeddings")
        except Exception as e:
            logger.error(f"Could not get collection for fallback: {e}")
            return "<NO CONTEXT AVAILABLE - COLLECTION ERROR>"

    try:
        all_nodes = collection.get(include=["metadatas", "documents"], limit=100)
        if not all_nodes["ids"]:
            return "<NO NODES AVAILABLE>"
        candidates = []
        for i in range(len(all_nodes["ids"])):
            node_id = all_nodes["ids"][i]
            doc = all_nodes["documents"][i]
            meta = all_nodes["metadatas"][i]

            if not doc or doc.startswith("<") or len(doc) < min_code_length:
                continue
            importance = 0.0
            if isinstance(meta, dict):
                if "combined" in meta:
                    importance = float(meta.get("combined", 0.0))
                elif "importance" in meta and isinstance(meta["importance"], dict):
                    importance = float(meta["importance"].get("combined", 0.0))
                kind = meta.get("kind", "")
                if kind == "CLASS":
                    importance += 2.0
                elif kind == "METHOD" and len(doc) > 200:
                    importance += 1.0
                elif "controller" in node_id.lower():
                    importance += 3.0
                elif "service" in node_id.lower():
                    importance += 2.0

            candidates.append((importance, node_id, doc, meta))

        if not candidates:
            return "<NO VALID CANDIDATES FOUND>"

        candidates.sort(key=lambda x: x[0], reverse=True)

        context_parts = []
        current_tokens = 0
        max_fallback_tokens = max_context_tokens // 2

        for importance, node_id, doc, meta in candidates[:10]:
            kind = meta.get("kind", "CODE")
            section = f"## {kind}: {node_id}\n{doc.strip()}"
            section_tokens = estimate_tokens(section, is_code=True)
            if current_tokens + section_tokens <= max_fallback_tokens:
                context_parts.append(section)
                current_tokens += section_tokens
            else:
                break

        result = "\n\n".join(context_parts) if context_parts else "<NO HIGH-IMPORTANCE NODES FOUND>"

        _context_cache['fallback'] = result
        _context_cache['timestamp'] = current_time

        logger.debug(f"Built fallback context: {current_tokens} tokens, {len(context_parts)} sections")
        return result

    except Exception as e:
        logger.error(f"Error building fallback context: {e}")
        error_context = f"<CONTEXT BUILD ERROR: {str(e)[:100]}>"
        _context_cache['fallback'] = error_context
        _context_cache['timestamp'] = current_time
        return error_context


def get_context_stats() -> Dict[str, Any]:
    """
        Returns statistics and state of the context cache.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - cache_stats (dict): Hit/miss counters.
                - cache_age (float): Seconds since last cache update.
                - has_cached_fallback (bool): Whether a cached fallback exists.
    """
    return {
        'cache_stats': _context_cache['stats'],
        'cache_age': time.time() - _context_cache['timestamp'],
        'has_cached_fallback': _context_cache['fallback'] is not None
    }


def clear_context_cache():
    """
        Clears the in-memory context cache and resets its statistics.

        Side Effects:
            Resets `_context_cache` fields: stored fallback, timestamp, and hit/miss stats.
    """
    global _context_cache
    _context_cache = {
        'fallback': None,
        'timestamp': 0,
        'stats': {'hits': 0, 'misses': 0}
    }
