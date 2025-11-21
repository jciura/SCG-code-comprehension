import os
import re
from typing import Optional


def extract_method_with_context(metadata: dict, context_lines: int = 5) -> str:
    """
    Extracts a method's source code along with surrounding context lines.

    Args:
        metadata: Node metadata containing 'uri' and 'location'
        context_lines: Number of context lines to include before and after

    Returns:
        Method code with line numbers and context, or error message
    """
    try:
        uri = metadata.get("uri", "")
        location = metadata.get("location", "")
        if not uri or not location:
            return ""
        if uri.startswith("file://"):
            uri = uri[7:]
        if not os.path.exists(uri):
            return ""
        start, _ = location.split(";")
        start_line, _ = map(int, start.split(":"))
        with open(uri, encoding="utf-8") as f:
            lines = f.readlines()
        if start_line > len(lines):
            return f"<Invalid start line: {start_line}>"
        start_idx = max(0, start_line - 1 - context_lines)
        open_braces = 0
        end_idx = start_line - 1
        method_started = False
        for i in range(start_line - 1, len(lines)):
            line = lines[i]
            open_braces += line.count("{") - line.count("}")
            if not method_started and "{" in line:
                method_started = True
            if method_started and open_braces == 0:
                end_idx = i
                break
        else:
            end_idx = min(len(lines) - 1, start_line + 20)

        end_idx = min(len(lines) - 1, end_idx + context_lines)
        context_lines_list = lines[start_idx : end_idx + 1]
        numbered_lines = []
        for i, line in enumerate(context_lines_list):
            line_num = start_idx + i + 1
            prefix = ">>>" if start_line <= line_num <= start_line + 2 else "   "
            numbered_lines.append(f"{prefix} {line_num:3d}: {line.rstrip()}")

        return "\n".join(numbered_lines)
    except Exception as e:
        return f"<Could not extract method with context: {e}>"


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
