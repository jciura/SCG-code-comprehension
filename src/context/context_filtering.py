import re

MAX_CODE_PREVIEW_CHARS = 2000
MAX_DEFINITION_LINES = 200
MAX_EXCEPTION_LINES = 150
MAX_TESTING_LINES = 300
MAX_IMPLEMENTATION_LINES = 350


def is_scala(node_id: str) -> bool:
    return "/" in node_id or node_id.endswith(".scala")

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
    if not code:
        return ""

    if kind not in ["CLASS", "INTERFACE", "METHOD"]:
        return code[:MAX_CODE_PREVIEW_CHARS]

    definition_lines = []
    lines = code.split("\n")
    modifiers = ["public ", "private ", "protected "]
    scala_mode = is_scala(node_id)

    if "." in node_id:
        class_name = node_id.split(".")[-1]
    elif "/" in node_id:
        class_name = node_id.split("/")[-1].replace("#", "")
        scala_mode = True
    else:
        class_name = node_id

    for line in lines:
        line_clean = line.strip()
        line_clean = re.sub(r'^/\*\*?\s*', '', line_clean)
        line_clean = re.sub(r'\*/', '', line_clean)
        if not line_clean or line_clean.startswith("//"):
            continue
        if line_clean.startswith("@"):
            definition_lines.append(line_clean)
            continue
        if ("class " in line_clean or "interface " in line_clean or
                "trait " in line_clean or "object " in line_clean):
            definition_lines.append(line_clean)
            continue
        if scala_mode:
            if re.search(r'\bdef\s+[\w<\[]+', line_clean) or line_clean.startswith("def this("):
                signature = line_clean.split("=")[0].strip()
                if not signature.endswith(";"):
                    signature += ";"
                definition_lines.append(signature)
                definition_lines.append(line_clean)
                continue
        else:
            has_modifier = any(line_clean.startswith(mod) for mod in modifiers)
            if not has_modifier:
                continue
            if "(" not in line_clean and "=" not in line_clean:
                definition_lines.append(line_clean)
                continue
            if class_name + "(" in line_clean:
                definition_lines.append(line_clean)
                continue
            if "(" in line_clean and ")" in line_clean:
                signature = line_clean.rstrip("{").strip()
                if not signature.endswith(";"):
                    signature += ";"
                definition_lines.append(signature)

    return "\n".join(definition_lines[:MAX_DEFINITION_LINES])


def filter_exception_code(code: str, node_id: str) -> str:
    """
    Extracts exception-related lines from code.

    Captures lines with `throw new`, `orElseThrow`, or words ending with
    'Exception' or 'Error', including relevant import statements.

    Args:
        code: Code text to analyze
        node_id: Identifier of the graph node (used to infer class name)

    Returns:
        Extracted exception-related lines joined as a string
    """
    if not code:
        return ""

    exception_lines = []
    lines = code.split("\n")
    scala_mode = is_scala(node_id)

    for line in lines:
        line_clean = line.strip()
        line_clean = re.sub(r'^/\*\*?\s*', '', line_clean)
        line_clean = re.sub(r'\*/', '', line_clean)
        if not line_clean or line_clean.startswith("//"):
            continue
        if "throw new" in line_clean or "orElseThrow(" in line_clean:
            exception_lines.append(line_clean)
            continue

        if scala_mode and "throw " in line_clean:
            exception_lines.append(line_clean)
            continue

        words = re.split(r"[().,;{}\s]+", line_clean)
        has_exception = False
        for word in words:
            if word.endswith(("Exception", "Error")) and len(word) > 5:
                has_exception = True
                break
        if has_exception:
            exception_lines.append(line_clean)
            continue
        if line_clean.startswith("import") and ("Exception" in line_clean or "Error" in line_clean):
            exception_lines.append(line_clean)
    return "\n".join(exception_lines[:MAX_EXCEPTION_LINES])


def filter_testing_code(code: str, node_id: str) -> str:
    """
    Filters and extracts testing-related code sections.

    Keeps test annotations, assertions, and mock setups.

    Args:
        code: Full code text to process
        node_id: Identifier of the graph node (used to infer class name)

    Returns:
        Filtered code snippet containing test-related elements
    """
    if not code:
        return ""
    testing_lines = []
    lines = code.split("\n")
    test_keywords = [
        "@Test",
        "@Before",
        "@After",
        "@Mock",
        "assert",
        "assertEquals",
        "assertTrue",
        "assertFalse",
        "assertNull",
        "assertNotNull",
        "assertThrows",
        "assertThat",
        "when(",
        "given(",
        "mock(",
        "should",
        "expect(",
    ]
    scala_mode = is_scala(node_id)

    if scala_mode:
        test_keywords.extend(["mustBe", "shouldBe", "shouldEqual", "contain", "intercept[", "assertResult"])

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        if any(keyword in line_stripped for keyword in test_keywords):
            testing_lines.append(line_stripped)
            continue

        if not scala_mode:
            if "void " in line_stripped and (
                    "test" in line_stripped.lower() or "should" in line_stripped.lower()):
                testing_lines.append(line_stripped)
        else:
            if line_stripped.startswith(("test(", "it should", "it must", "describe(")):
                testing_lines.append(line_stripped)
            elif "def " in line_stripped and ("test" in line_stripped.lower() or "should" in line_stripped.lower()):
                testing_lines.append(line_stripped)

    return "\n".join(testing_lines[:MAX_TESTING_LINES])


def filter_implementation_code(code: str, node_id: str) -> str:
    """
    Filters code to show implementation details, removing boilerplate.

    Removes simple getters/setters and keeps business logic.

    Args:
        code: Full code text to process
        node_id: Identifier of the graph node (used to infer class name)

    Returns:
        Filtered code snippet containing implementation details
    """
    if not code:
        return ""

    implementation_lines = []
    scala_mode = is_scala(node_id)
    lines = code.split("\n")

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        if not scala_mode:
            if line_stripped.startswith(("public ", "private ", "protected ")):
                lower_line = line_stripped.lower()
                if " get" in lower_line and "()" in line_stripped and len(line_stripped) < 50:
                    continue
                if " set" in lower_line and len(line_stripped) < 60:
                    continue
            if line_stripped.startswith("return this.") and line_stripped.endswith(";"):
                if len(line_stripped) < 30:
                    continue
            if line_stripped.startswith("this.") and "=" in line_stripped and len(line_stripped) < 40:
                continue
        else:
            if re.match(r'^(override\s+)?(val|var)\s+\w+\s*:\s*\w+\s*=\s*\w+$', line_stripped):
                if len(line_stripped) < 40:
                    continue

        implementation_lines.append(line_stripped)

    return "\n".join(implementation_lines[:MAX_IMPLEMENTATION_LINES])
