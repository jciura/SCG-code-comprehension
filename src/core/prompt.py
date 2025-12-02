from typing import Any, Dict

from loguru import logger

from src.core.intent_analyzer import IntentCategory, get_intent_analyzer
from src.core.models import ConversationHistory


def get_task_instructions(intent_category: IntentCategory) -> str:
    """
    Returns task-specific instructions based on intent category.

    Args:
        intent_category: The detected intent category

    Returns:
        str: Task instructions for the LLM
    """
    if intent_category == IntentCategory.USAGE:
        return """Find where the requested code element is used in the codebase.
        Your task is to: identify all places where this element is called or referenced.

        Rules:
        1. Quote only exact class names, method names visible in the code below
        2. If you see a test method calling it, give its exact name from the code
        3. If you see a controller using it, give the exact @Mapping annotation
        4. Do not invent test names, method names, or class names
        5. Do not say "there are tests" if you don't see the exact test code
        6. If the element is not used anywhere in the code provided, say:
            "The code provided does not show where this method is used."
        7. Do not add line numbers (they are not in the code) 
        8. Do not add "at line X"

        What to report (only if visible in code):
        - Exact class name using it (copy from code)
        - Exact method name calling it (copy from code)
        - Exact HTTP mapping if present (copy annotation)
        - Line numbers if available

        Never make up test names or usage locations. Base everything on the code below."""

    elif intent_category == IntentCategory.DEFINITION:
        return """Describe the code element in a clear, natural way based strictly 
                    on the provided code.

        Rules:
        1) Base your answer only on what is visible in the code below
        2) Use exact names and types from the code
        3) Do not explain what annotations/frameworks do
        4) Do not add knowledge not in the code
        5) Do not mention @Table if you don't see @Table in the code


        Answer structure:

        Opening (1-2 sentences):
        - Identify what kind of element this is (class, interface, method, enum, etc.)
        - State its main purpose or what it represents

        Main content (adapt based on element type):

        For CLASSES/INTERFACES:
        - List main fields with types (e.g., "contains three fields: id, name, and items")
        - Mention relationships (e.g., "has a many-to-many relationship with X")
        - Group methods logically: "provides constructors", "has getters/setters", 
            "includes validation methods"

        For METHODS:
        - State what parameters it takes
        - Describe what it does (based on visible code)
        - Mention what it returns
        - Note any exceptions it throws

        For ENUMS:
        - List the enum values
        - Mention any fields or methods if present

        For ANNOTATIONS/INTERFACES:
        - State its purpose
        - List methods/fields if applicable


        Style:
        -Group similar items ("has getters and setters for all fields")
        - Use natural language
        -Be specific with types ("Set<Webinar>", "int categoryId")
        - Mention key annotations without explaining them
        - Don't list every single method individually
        - Don't explain what frameworks/annotations do
        - Don't add bullet-point lists for simple things
        - Don't copy-paste entire signatures


        Adapt to the code:
        - If code shows only fields, focus on fields
        - If code shows only method signatures, describe methods
        - If code shows full implementation, describe behavior
        - If element is simple, keep answer short
        - If element is complex, add more detail

        Base your answer only on the code below."""

    elif intent_category == IntentCategory.IMPLEMENTATION:
        return """Explain how the code works internally, using only what is visible below.

        Rules:
        1) Describe only what the code shows, no assumptions, no theory.
        2) Use exact method/variable names and control flow as written.
        3) Do not explain general concepts or frameworks.

        Output format:
        - Responsibilities: one short paragraph summarizing what the code does (only from code).
        - Data flow: step-by-step of how inputs become outputs (variables/methods named exactly).
        - Key methods and logic: for each important method, list the main steps, conditions, 
            and returned values (exact syntax where helpful).

        Avoid:
        - Any “why” or best practices.
        - Any behavior not literally present in the code.

        Answer only based on the provided code."""

    elif intent_category == IntentCategory.TESTING:
        return """Describe the testing code strictly based on what is visible below.

        Rules:
        1) Mention only test classes and methods present in the code.
        2) Quote exact test names, methods, and assertion calls.
        3) Do not explain testing concepts, frameworks, or best practices.
        4) Base everything strictly on the provided code — no assumptions.

        Focus your answer on:
        - Which methods or behaviors are tested (from visible code)
        - The exact test method names
        - The assertions used (copy their syntax)
        - Any setup or mock usage, if explicitly visible

        Write a clear, factual summary, no explanations or suggestions."""

    elif intent_category == IntentCategory.EXCEPTION:
        return """Identify all exception handling visible in the provided code.

        Rules:
        1) Mention only exceptions that appear explicitly in the code 
            (try-catch, throws, or class references).
        2) Use exact exception class names and method names as written.
        3) Do not explain what the exceptions mean or how they work.
        4) Base everything strictly on what is visible in the code.

        Focus your answer on:
        - Which exceptions are caught or thrown (exact names)
        - Where they appear (methods, blocks, annotations)
        - How they are handled or propagated (visible code only)

        If no exceptions appear in the code, simply state that none are present."""

    elif intent_category == IntentCategory.TOP:
        return """List the most relevant code elements found in the provided context.

        Rules:
        1) Return only names of classes or methods as they appear in the code.
        2) Keep the same order as in the context.
        3) Number each item (e.g., 1., 2., 3.) to each item add parameter 
            attached to them in context.
        4) Do not add explanations, summaries, or descriptions.
        5) If no names are visible, return: "<NO NAMES FOUND>".
        6) Skip: Test classes, Exceptions, DTOs, Configs" 

        """

    else:
        return """Analyze the provided code and describe what is visible in it.

        Rules:
        1) Base your answer only on the code below, no assumptions or external knowledge.
        2) Use exact class, method, and variable names as they appear.
        3) Do not explain programming concepts, frameworks, or best practices.
        4) Mention only what is explicitly shown in the code.

        Focus your answer on:
        - The main classes, interfaces, or functions defined
        - Key fields or variables and their types
        - Important methods and their purpose based on visible logic
        - Any relationships, annotations, or interactions seen in the code

        Keep the tone factual and concise, describe only what you see."""


def build_prompt(
    question: str, context: str, intent: Dict[str, Any], conversation_history: ConversationHistory
) -> str:
    """
    Builds an intent-aware prompt from code context and conversation.

    Args:
        question: User question
        context: Code/context to analyze (may be truncated)
        intent: Intent payload with at least "primary_intent"
        conversation_history: Rolling chat history

    Returns:
        str: Fully formatted prompt string ready for the LLM
    """
    primary_intent = intent["primary_intent"]
    try:
        intent_category = IntentCategory(primary_intent)
    except ValueError:
        intent_category = IntentCategory.GENERAL

    task = get_task_instructions(intent_category)
    analyzer = get_intent_analyzer()
    limits = analyzer.get_context_limits(intent_category)
    max_context_chars = limits["max_context_chars"]
    if intent_category == IntentCategory.TOP:
        return "\n".join(
            [
                f"INSTRUCTIONS: {task}",
                f"USER QUESTION: {question}",
                f"CONTEXT NAMES: {context.strip() or '<NO NAMES FOUND>'}",
            ]
        )
    context = _truncate_context(context, max_context_chars)
    prompt_parts = [
        f"INSTRUCTIONS: {task}",
        "",
        "CODE TO ANALYZE:",
        context.strip(),
        "",
        f"USER QUESTION: {question}",
        "",
        "YOUR ANSWER (extract facts from code above):",
    ]
    if intent_category not in [IntentCategory.EXCEPTION, IntentCategory.TESTING]:
        history_context = conversation_history.get_conversation_context()
        if history_context and len(history_context) < 500:
            prompt_parts.insert(-2, f"Previous conversation: {history_context[:300]}")
            prompt_parts.insert(-2, "")

    final_prompt = "\n".join(prompt_parts)
    logger.debug(f"Context length: {len(context)} chars")
    logger.debug(f"Has ## headers: {context.count('##')} sections")
    logger.debug(f"Final prompt length: {len(final_prompt)} chars")

    return final_prompt


def _truncate_context(context: str, max_chars: int) -> str:
    """
    Truncates context to maximum character limit while preserving section structure.

    Args:
        context: The context string to truncate
        max_chars: Maximum allowed characters

    Returns:
        str: Truncated context
    """
    if len(context) <= max_chars:
        return context

    if "##" in context:
        sections = context.split("\n\n##")
        truncated = []
        current_length = 0

        for i, section in enumerate(sections):
            section_text = section if i == 0 else "##" + section

            if current_length + len(section_text) + 2 <= max_chars:
                truncated.append(section_text)
                current_length += len(section_text) + 2
            else:
                break

        if truncated:
            result = "\n\n".join(truncated)
            if len(truncated) < len(sections):
                result += f"\n\n... [{len(sections) - len(truncated)} sections omitted]"
            return result

    return context[:max_chars] + "\n... [truncated]"
