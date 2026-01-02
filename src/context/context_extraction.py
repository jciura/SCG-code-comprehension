import re
from typing import List, Optional


def extract_usage_fragment(code: str, target: str, context_lines: int = 5) -> List[str]:
    """
    Extracts a fragment of code showing usage of a target method or class.

    Args:
        code: Source code text
        target: Method or class name to locate
        context_lines: Number of lines before and after to include

    Returns:
        Code fragment containing the usage, or None if not found
    """
    if not target or not code:
        return None
    pattern = re.compile(rf"\b{re.escape(target)}\b", re.IGNORECASE)
    code_lines = code.split("\n")
    fragments = []
    used_lines = set()
    for i, line in enumerate(code_lines):
        if pattern.search(line):
            start = max(0, i - context_lines)
            end = min(len(code_lines), i + context_lines + 1)
            if any(start <= line <= end for line in used_lines):
                continue
            fragments.append("\n".join(code_lines[start:end]))
            used_lines.update(range(start, end))

    return fragments


def extract_target_from_question(question: str) -> Optional[str]:
    """
    Extracts a probable target entity (method/class) name from a question.

    Args:
        question: Natural language question text

    Returns:
        Extracted target name, or None if no match found
    """
    if not question:
        return None
    for pattern in [r'\b([A-Z][a-zA-Z]*[a-z][a-zA-Z]*)\b', r'\b([a-z]+[A-Z][a-zA-Z]*)\b',]:
        match = re.search(pattern, question)
        if match:
            return match.group(1)
    question_lower = question.lower()
    patterns = [
        r"(?:method|function|class|trait|object|interface|def|type)\s+(\w+)",
        r"(\w+)\s+(?:method|function|class|trait|object|interface)",
        r"(?:where\s+is|how\s+to\s+use|what\s+is|what\s+does)\s+(\w+)",
        r"tests?\s+(?:for\s+)?(\w+)",
        r"implementation\s+of\s+(\w+)",
        r"(\w+)\s+implementation",
        r"(\w+(?:controller|service|repository|handler|factory|builder|utils?|helper|exception|error))\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, question_lower)
        if match:
            return match.group(1)
    return None