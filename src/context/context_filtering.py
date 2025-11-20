import re


def filter_definition_code(code: str, node_id: str, kind: str) -> str:
    """
    Filters and summarizes definition-related code sections.

    Keeps only key structural elements such as class/interface declarations,
    field signatures, constructors, and method headers.

    Args:
        code: Full code text to process
        node_id: Identifier of the graph node (used to infer class name)
        kind: Node type, e.g. "CLASS" or "INTERFACE"

    Returns:
        Filtered code snippet containing main definitions
    """
    if not code or kind not in ["CLASS", "INTERFACE"]:
        return code[:400] if code else ""

    definition_lines = []
    lines = code.split("\n")
    for line in lines:
        line_clean = line.strip()
        if not line_clean or line_clean.startswith("//") or line_clean.startswith("*"):
            continue
        if (
            line_clean.startswith("@")
            or line_clean.startswith("public class")
            or line_clean.startswith("public interface")
            or line_clean.startswith("private class")
            or line_clean.startswith("protected class")
        ):
            definition_lines.append(line_clean)
            continue
        if (
            (
                line_clean.startswith("private ")
                or line_clean.startswith("protected ")
                or line_clean.startswith("public ")
            )
            and "(" not in line_clean
            and "=" not in line_clean
        ):
            definition_lines.append(line_clean)
            continue
        class_name = node_id.split(".")[-1] if "." in node_id else node_id
        if (
            ("public " + class_name + "(") in line_clean
            or ("private " + class_name + "(") in line_clean
            or ("protected " + class_name + "(") in line_clean
        ):
            definition_lines.append(line_clean)
            continue
        if (
            any(modifier in line_clean for modifier in ["public ", "private ", "protected "])
            and "(" in line_clean
            and ")" in line_clean
            and not line_clean.startswith("@")
        ):
            method_signature = (
                line_clean[:-1].strip() + ";" if line_clean.endswith("{") else line_clean + ";"
            )
            definition_lines.append(method_signature)
            continue
        if line_clean == "}" and len(definition_lines) > 5:
            definition_lines.append(line_clean)
            break

    return "\n".join(definition_lines[:15])


def filter_exception_code(code: str) -> str:
    """
    Extracts exception-related lines from code.

    Captures lines with `throw new`, `orElseThrow`, or words ending with
    'Exception' or 'Error', including relevant import statements.

    Args:
        code: Code text to analyze

    Returns:
        Extracted exception-related lines joined as a string
    """
    if not code:
        return ""

    exception_lines = []
    lines = code.split("\n")
    for line in lines:
        line_clean = line.strip()
        if "throw new" in line_clean or "orElseThrow(" in line_clean:
            exception_lines.append(line_clean)
            continue
        words = re.split(r"[().,;{}\s]+", line_clean)
        has_exception = any(
            word.endswith(("Exception", "Error")) and len(word) > 5 for word in words
        )
        if has_exception:
            exception_lines.append(line_clean)
            continue
        if line_clean.startswith("import") and ("Exception" in line_clean or "Error" in line_clean):
            exception_lines.append(line_clean)
    return "\n".join(exception_lines[:40])
