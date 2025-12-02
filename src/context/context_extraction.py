import re
from typing import Optional


def extract_usage_fragment(code: str, target_method: str, context_lines: int = 5) -> Optional[str]:
    """
    Extracts a fragment of code showing usage of a target method.

    Args:
        code: Source code text
        target_method: Method name to locate
        context_lines: Number of lines before and after to include

    Returns:
        Code fragment containing the method call, or None if not found
    """
    if not target_method or f"{target_method}(" not in code:
        return None
    code_lines = code.split("\n")
    for i, line in enumerate(code_lines):
        if f"{target_method}(" in line:
            start = max(0, i - context_lines)
            end = min(len(code_lines), i + context_lines + 1)
            return "\n".join(code_lines[start:end])
    return None


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

    question_lower = question.lower()
    patterns = [
        r"method\s+(\w+)",
        r"(\w+)\s+method",
        r"class\s+(\w+)",
        r"(\w+)\s+class",
        r"for\s+(\w+)\s+class",
        r"where.*\s+(\w+)\s+used",
        r"tests?\s+for\s+(\w+)",
        r"(\w+controller)",
        r"(\w+service)",
        r"(\w+repository)",
    ]

    for pattern in patterns:
        match = re.search(pattern, question_lower)
        if match:
            return match.group(1)
    return None
