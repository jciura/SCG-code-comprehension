from typing import Any, Dict

from loguru import logger

from src.core.intent_analyzer import IntentCategory


def get_task_instructions(intent_category: IntentCategory) -> str:
    """
    Returns task-specific instructions based on intent category.

    Args:
        intent_category: The detected intent category

    Returns:
        str: Task instructions for the LLM
    """
    base_role = """You are a senior software engineer with over 10 years of experience
    working on large-scale codebases.

    You specialize in:
    - reading unfamiliar code accurately
    - extracting only what is explicitly visible
    - answering with precision

    You do not guess or assume.
    The provided code is your only source of truth.
    """

    if intent_category == IntentCategory.USAGE:
        return f"""{base_role}
    Task:
    Find where the requested code element is used in the provided code.

    Rules:
    1) Use only identifiers exactly as they appear 
     in the code (class names, method names, annotations).
     Do not rephrase them.
    2) Report usage only if the call/reference is explicitly visible in the code.
    3) If a test calls it, report the exact test class and exact test method name (only if visible).
    4) If a controller uses it, report the exact HTTP mapping annotation.
    5) Do not invent test names, method names, class names, or usage locations.
    6) Do not generalize (e.g., "there are tests") unless you see the exact test code.
    7) Do not add line numbers unless line numbers are explicitly present.
    8) If no usage is visible, respond with:
    "The code provided does not show where this method is used."

    Output:
    - Class name or names using it
    - Method name or names calling/referencing it
    - HTTP mapping annotations, if present
    (Only include items that are visible in the code.)
    """

    elif intent_category == IntentCategory.DEFINITION:
        return f"""{base_role}
    Task:
    Describe the requested code element clearly and factually using only the provided code.
    Rules:
    1) Base your answer only on what is visible in the code.
    2) Use exact names, types, and identifiers as written.
    3) Mention annotations only if they appear in the code.
    4) Do not explain what frameworks/annotations do.
    5) Do not add external knowledge, assumptions.


    Required structure:
    Opening (1–2 sentences):
    - Identify what the element is (class/interface/method/enum/etc.).
    - State its role based on visible code.

    Main content (adapt to element type):
    - Classes/Interfaces: fields (with types), visible relationships, grouped method summaries.
    - Methods: parameters, visible behavior, return value, thrown exceptions (if visible).
    - Enums: values, fields/methods if present.
    - Annotations/Interfaces: purpose as visible + members if present.
    """

    elif intent_category == IntentCategory.IMPLEMENTATION:
        return f"""{base_role}

    Task:
    Explain how the code works internally, strictly based on visible implementation.

    Rules:
    1) Describe only behavior that is present in the code.
    2) Do not explain general concepts, patterns, or frameworks.
    3) Do not speculate about intent or design decisions.

    Output format:
    Responsibilities:
    - One short paragraph summarizing what the code does (from code only).

    Data flow:
    - Step-by-step: how inputs are transformed into outputs (use exact variable/method names).

    Key methods and logic:
    - For each important method: main steps, conditions/branches, returned values (as visible).
    """

    elif intent_category == IntentCategory.TESTING:
        return f"""{base_role}

    Task:
    Summarize the testing code based on what is visible.

    Rules:
    1) Mention only test classes and test methods present in the provided code.
    2) Quote exact test method names and exact assertion calls as written.
    3) Do not explain testing concepts, frameworks, or best practices.
    4) Do not infer what is tested beyond what the code explicitly asserts.

    Focus on:
    - Which methods/behaviors are tested (based on visible calls/assertions)
    - Exact test class names
    - Exact test method names
    - Assertions used 

    Output:
    - A clear, factual summary.
    """

    elif intent_category == IntentCategory.EXCEPTION:
        return f"""{base_role}

    Task:
    Identify all exception handling visible in the provided code.

    Rules:
    1) Mention only exceptions that appear explicitly (try/catch, throws, or direct references).
    2) Use exact exception class names and exact method names where they appear.

    Output:
    - Exceptions caught (with the surrounding method/block name if visible)
    - Exceptions thrown (throws ...)
    - How they are handled (only what is shown)

    If no exceptions appear in the code, respond:
    "No exception handling is visible in the provided code."
    """
    elif intent_category == IntentCategory.TOP:
        return f"""{base_role}

    Task:
    List the most relevant code element names found in the provided context.

    Rules:
    1) Return only names of classes or methods as they appear in the context.
    2) Keep the same order as they appear in the context.
    3) Number each item (1., 2., 3., ...). For each item include 
        any parameters attached to it in the context.
    4) If no names are visible, return exactly: "<NO NAMES FOUND>"

    Output:
    - A numbered list of names only.
    
    RULES: DON'T ASK THE SAME QUESTION IF YOU GOT ANSWER. JUST LIST NODES THAT YOU GET.
    """

    else:
        return f"""{base_role}
    
        Task:
        Analyze the provided code and describe what is visible in it.
    
        Rules:
        1) Base your answer only on the code below (no external knowledge).
        2) Use exact class, method, and variable names as they appear.
        3) Do not explain programming concepts, frameworks.
        4) Mention only what is explicitly shown.
    
        Focus on:
        - Main classes, interfaces,functions defined
        - Key fields, variables and their types
        - Important methods
        - Visible relationships, annotations, interactions
        """


def build_prompt(question: str, context: str, intent: Dict[str, Any]) -> str:
    """
    Builds an intent-aware prompt from code context and conversation.

    Args:
        question: User question
        context: Code/context to analyze (may be truncated)
        intent: Intent payload with at least "primary_intent"

    Returns:
        str: Fully formatted prompt string ready for the LLM
    """
    primary_intent = intent["primary_intent"]
    try:
        intent_category = IntentCategory(primary_intent)
    except ValueError:
        intent_category = IntentCategory.GENERAL

    task = get_task_instructions(intent_category)
    if intent_category == IntentCategory.TOP:
        return "\n".join(
            [
                f"Instructions:\n{task}",
                "",
                f"User question:\n{question}",
                "",
                f"Context names:\n{context.strip() or '<NO NAMES FOUND>'}",
            ]
        )

    prompt_parts = f"""Instructions: {task}
    Code to analyze: {context.strip()}
    User question: {question}
    Your answer:"""

    final_prompt = prompt_parts
    logger.debug(f"Context length: {len(context)} chars")
    logger.debug(f"Final prompt length: {len(final_prompt)} chars")

    return final_prompt
