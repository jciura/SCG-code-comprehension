from typing import List, Tuple, Dict, Any, Set, Optional
import time
import json
import re
from collections import Counter


max_context_tokens = 2000
avg_chars_per_token = 3.8
code_chars_per_token = 3.5
text_chars_per_token = 4.2
min_code_length = 30
fallback_cache_duration = 300
max_sections_per_category = 8


_context_cache = {
    'fallback': None,
    'timestamp': 0,
    'stats': {'hits': 0, 'misses': 0}
}

def filter_definition_code(code: str, node_id: str, kind: str) -> str:
    if not code or kind not in ["CLASS", "INTERFACE"]:
        return code[:400]

    definition_lines = []
    in_method_body = False
    brace_count = 0

    for line in code.split('\n'):
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
            if line_clean.endswith('{'):
                method_signature = line_clean[:-1].strip() + ';'
            else:
                method_signature = line_clean + ';'
            definition_lines.append(method_signature)
            continue

        if line_clean == '}' and len(definition_lines) > 5:
            definition_lines.append(line_clean)
            break

    return '\n'.join(definition_lines[:15])

def estimate_tokens(text: str, is_code: bool = True) -> int:
    if not text:
        return 0

    chars_per_token = code_chars_per_token if is_code else text_chars_per_token
    non_whitespace_chars = len(text.replace(' ', '').replace('\n', '').replace('\t', ''))
    if len(text) > 0:
        whitespace_ratio = 1 - (non_whitespace_chars / len(text))
    else:
        whitespace_ratio = 0
    adjusted_ratio = chars_per_token * (1 + whitespace_ratio * 0.3)
    return max(1, int(len(text) / adjusted_ratio))


def get_node_priority_score(node_data: Dict[str, Any], category: str) -> float:
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
        elif "public class" in code:
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


def filter_exception_code(code: str) -> str:
    if not code:
        return ""
    exception_lines = []
    lines = code.split('\n')
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if 'throw new' in line_clean or 'orElseThrow(' in line_clean:
            exception_lines.append(line_clean)
            continue
        words = re.split(r'[().,;{}\s]+', line_clean)
        has_exception = any(word.endswith(('Exception', 'Error')) and len(word) > 5
                            for word in words)
        if has_exception:
            exception_lines.append(line_clean)
            continue

        if line_clean.startswith('import') and ('Exception' in line_clean or 'Error' in line_clean):
            exception_lines.append(line_clean)



    return '\n'.join(exception_lines[:40])


def build_context(nodes: List[Tuple[float, Dict[str, Any]]], category: str, confidence: float) -> str:
    if not nodes:
        return build_fallback_context()
    if category == "definition" and nodes:
        best_node = nodes[0][1]
        code = best_node.get("code", "")
        node_id = best_node.get("node", "")
        kind = best_node.get("metadata", {}).get("kind", "")

        if kind == "CLASS" and code:
            important_lines = []
            for line in code.split('\n'):
                line_clean = line.strip()
                if (line_clean.startswith('@') or
                        'class ' in line_clean or
                        (line_clean.startswith(('private ', 'public ', 'protected ')) and
                         '(' not in line_clean and ';' in line_clean)):
                    important_lines.append(line_clean)
                if (any(mod in line_clean for mod in ['public ', 'private ', 'protected ']) and
                        '(' in line_clean and ')' in line_clean and
                        not line_clean.startswith('@')):
                    break
                if len(important_lines) > 10:
                    break
            filtered_code = '\n'.join(important_lines)
            result = f"## {kind}: {node_id}\n{filtered_code}"
            return result
    if category in ["medium", "general"] and confidence < 0.7:
        if nodes and len(nodes) > 0:
            first_node = nodes[0][1]
            metadata = first_node.get("metadata", {})
            node_kind = metadata.get("kind", "")

            if node_kind in ["CLASS", "INTERFACE"]:
                category = "definition"
                confidence = 0.8
            elif node_kind == "METHOD":
                category = "implementation"
                confidence = 0.7

    context_limits = {
        "exception": {
            "max_tokens": 1000,
            "base_nodes":3,
            "category_nodes": 1,
            "fill_nodes": 0
        },
        "testing": {
            "max_tokens": 300,
            "base_nodes": 1,
            "category_nodes": 0,
            "fill_nodes": 0
        },
        "usage": {
            "max_tokens": 400,
            "base_nodes": 1,
            "category_nodes": 1,
            "fill_nodes": 0
        },
        "definition": {
            "max_tokens": 250,
            "base_nodes": 1,
            "category_nodes": 0,
            "fill_nodes": 0
        },
        "implementation": {
            "max_tokens": 500,
            "base_nodes": 1,
            "category_nodes": 1,
            "fill_nodes": 0
        }
    }

    limits = context_limits.get(category, {
        "max_tokens": 600,
        "base_nodes": 2,
        "category_nodes": 1,
        "fill_nodes": 1
    })

    MAX_CONTEXT_TOKENS_LOCAL = limits["max_tokens"]
    base_nodes = limits["base_nodes"]
    category_nodes = limits["category_nodes"]
    fill_nodes = limits["fill_nodes"]

    print(f"Building context: category={category}, confidence={confidence:.2f}, "
          f"nodes={len(nodes)}, limit={MAX_CONTEXT_TOKENS_LOCAL}")

    context_sections = []
    current_tokens = 0
    seen_codes: Set[str] = set()
    used_nodes: Set[str] = set()
    section_counts = Counter()

    def add_node_section(node_data: Dict[str, Any], section_label: str, priority: int = 0) -> bool:
        nonlocal current_tokens
        code = node_data.get("code", "")
        node_id = node_data.get("node", "")
        kind = node_data.get("metadata", {}).get("kind", "CODE")
        if (not code or
                len(code) < 20 or
                code in seen_codes or
                node_id in used_nodes or
                code.startswith('<')):
            return False

        if section_counts[section_label] >= 2:
            return False
        if category == "exception":
            filtered_code = filter_exception_code(code)
            if not filtered_code:
                print(f"Skipping {node_id} - no exception handling found")
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
            max_code_length = 150 if category in ["usage", "testing"] else 200
            if len(code) > max_code_length:
                code_preview = code[:max_code_length] + "..."
            else:
                code_preview = code
            header = f"## {kind}: {node_id}\n"

        full_section = header + code_preview + "\n"
        section_tokens = estimate_tokens(full_section, is_code=True)
        buffer_tokens = 10
        if current_tokens + section_tokens <= MAX_CONTEXT_TOKENS_LOCAL - buffer_tokens:
            context_sections.append((priority, full_section))
            seen_codes.add(code)
            used_nodes.add(node_id)
            current_tokens += section_tokens
            section_counts[section_label] += 1
            return True

        return False

    if confidence >= 0.8:
        base_nodes = min(base_nodes + 1, 2)
    elif confidence < 0.5:
        base_nodes = max(base_nodes - 1, 1)

    scored_nodes = []
    for score, node_data in nodes:
        priority_score = get_node_priority_score(node_data, category)
        combined_score = score + priority_score * 0.1
        scored_nodes.append((combined_score, node_data))

    scored_nodes.sort(key=lambda x: x[0], reverse=True)
    added_base = 0
    for score, node_data in scored_nodes[:base_nodes * 2]:
        if added_base >= base_nodes:
            break
        if add_node_section(node_data, "top-match", priority=100):
            added_base += 1

    print(f"Added {added_base} base nodes")

    if category_nodes > 0 and current_tokens < MAX_CONTEXT_TOKENS_LOCAL * 0.8:
        remaining_nodes = [(s, n) for s, n in scored_nodes if n.get("node", "") not in used_nodes]
        added_category = 0

        if category == "usage":
            for score, node_data in remaining_nodes:
                if added_category >= category_nodes:
                    break
                node_id = node_data.get("node", "").lower()
                code = node_data.get("code", "")
                if (("controller" in node_id and "test" not in node_id) or
                        any(annotation in code for annotation in
                            ["@GetMapping", "@PostMapping", "@PutMapping", "@DeleteMapping"]) or
                        ("service" in node_id and "@Autowired" in code)):
                    if add_node_section(node_data, "usage-pattern", priority=80):
                        added_category += 1

        elif category == "implementation":
            for score, node_data in remaining_nodes:
                if added_category >= category_nodes:
                    break

                metadata = node_data.get("metadata", {})
                kind = metadata.get("kind", "")
                code = node_data.get("code", "")

                if ((kind == "METHOD" and len(code) > 100) or
                        kind == "CONSTRUCTOR" or
                        "algorithm" in code.lower()):
                    if add_node_section(node_data, "implementation", priority=75):
                        added_category += 1

        print(f"Added {added_category} category-specific nodes")
    if fill_nodes > 0 and current_tokens < MAX_CONTEXT_TOKENS_LOCAL * 0.6:
        final_remaining = [(s, n) for s, n in scored_nodes if n.get("node", "") not in used_nodes]
        added_fill = 0
        for score, node_data in final_remaining[:fill_nodes]:
            if current_tokens >= MAX_CONTEXT_TOKENS_LOCAL * 0.9:
                break
            if added_fill >= fill_nodes:
                break
            if add_node_section(node_data, "additional", priority=50):
                added_fill += 1
        print(f"Added {added_fill} fill nodes")

    if not context_sections:
        return build_fallback_context()

    context_sections.sort(key=lambda x: x[0], reverse=True)
    final_context = "\n".join(section[1] for section in context_sections)

    if len(final_context) > MAX_CONTEXT_TOKENS_LOCAL * 4:
        lines = final_context.split('\n')
        essential_lines = [line for line in lines[:15] if line.strip()]
        final_context = '\n'.join(essential_lines)


    print(f"Final context: {current_tokens} tokens, {len(context_sections)} sections, {len(final_context)} chars")

    return final_context

def build_fallback_context(collection=None) -> str:
    global _context_cache
    current_time = time.time()
    if (_context_cache['fallback'] is not None and
            current_time - _context_cache['timestamp'] < fallback_cache_duration):
        _context_cache['stats']['hits'] += 1
        print("Using cached fallback context")
        return _context_cache['fallback']

    _context_cache['stats']['misses'] += 1
    if collection is None:
        try:
            from graph.retriver import chroma_client
            collection = chroma_client.get_collection(name="scg_embeddings")
        except Exception as e:
            print(f"Could not get collection for fallback: {e}")
            return "<NO CONTEXT AVAILABLE - COLLECTION ERROR>"

    try:
        # print("Building new fallback context...")
        all_nodes = collection.get(
            include=["metadatas", "documents", "ids"],
            limit=100
        )
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
            header = f"## {kind}: {node_id}\n"
            section = header + doc.strip()

            section_tokens = estimate_tokens(section, is_code=True)
            if current_tokens + section_tokens <= max_fallback_tokens:
                context_parts.append(section)
                current_tokens += section_tokens
            else:
                break

        result = "\n\n".join(context_parts) if context_parts else "<NO HIGH-IMPORTANCE NODES FOUND>"

        _context_cache['fallback'] = result
        _context_cache['timestamp'] = current_time

        print(f"Built fallback context: {current_tokens} tokens, {len(context_parts)} sections")
        return result

    except Exception as e:
        print(f"Error building fallback context: {e}")
        error_context = f"<CONTEXT BUILD ERROR: {str(e)[:100]}>"

        _context_cache['fallback'] = error_context
        _context_cache['timestamp'] = current_time

        return error_context


def get_context_stats() -> Dict[str, Any]:
    return {
        'cache_stats': _context_cache['stats'],
        'cache_age': time.time() - _context_cache['timestamp'],
        'has_cached_fallback': _context_cache['fallback'] is not None
    }


def clear_context_cache():
    global _context_cache
    _context_cache = {
        'fallback': None,
        'timestamp': 0,
        'stats': {'hits': 0, 'misses': 0}
    }
