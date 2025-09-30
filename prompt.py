from typing import Dict, Any
from models import ConversationHistory


def analyze_user_intent(question: str) -> Dict[str, Any]:
    question = question.lower()
    intent = {
        "primary_intent": "general",
        "secondary_intents": [],
        "requires_examples": False,
        "requires_usage_info": False,
        "requires_implementation_details": False,
        "user_expertise_level": "intermediate"
    }

    if any(word in question for word in ['used', 'where', 'usage', 'called', 'references', 'invoked']):
        intent["primary_intent"] = "usage"
        intent["requires_usage_info"] = True
    elif any(word in question for word in ['what is', 'describe', 'explain', 'define', 'meaning']):
        intent["primary_intent"] = "definition"
        intent["requires_examples"] = True
    elif any(word in question for word in ['how does', 'implementation', 'algorithm', 'logic', 'works']):
        intent["primary_intent"] = "implementation"
        intent["requires_implementation_details"] = True
    elif any(word in question for word in ['test', 'testing', 'junit', 'mock', 'verify']):
        intent["primary_intent"] = "testing"
        intent["requires_examples"] = True
    elif any(word in question for word in ['error', 'exception', 'throw', 'catch', 'fail']):
        intent["primary_intent"] = "exception"
        intent["requires_examples"] = True

    if any(word in question for word in ['example', 'sample', 'show me']):
        intent["secondary_intents"].append("examples")
        intent["requires_examples"] = True

    if any(word in question for word in ['best practice', 'recommendation', 'should i']):
        intent["secondary_intents"].append("advice")

    if any(word in question for word in ['why', 'reason', 'purpose']):
        intent["secondary_intents"].append("reasoning")

    return intent


def build_intent_aware_prompt(question: str, context: str, intent: Dict[str, Any],
                              conversation_history: ConversationHistory) -> str:
    primary_intent = intent["primary_intent"]
    if primary_intent == "exception":
        task = "Find error handling and exception management in this code."
    elif primary_intent == "usage":
        task = "Show where and how this code element is used."
    elif primary_intent == "definition":
        task = "Explain what this code element is and its purpose."
    elif primary_intent == "implementation":
        task = "Explain how this code works internally."
    else:
        task = "Analyze the provided code."

    max_context_chars = {
        "exception": 6000,
        "testing": 6000,
        "usage": 3000,
        "definition": 3000,
        "implementation": 4000,
        "general": 3000
    }.get(primary_intent, 1000)

    if len(context) > max_context_chars:
        sections = context.split('##')
        if len(sections) > 1:
            context = '##' + sections[1][:max_context_chars]
        else:
            context = context[:max_context_chars]
        context += "\n... [content truncated for brevity]"

    prompt_parts = [
        f"Task: {task}",
        "",
        "Code:",
        context,
        "",
        f"Question: {question}",
        "",
        "Answer:"
    ]

    if primary_intent not in ["exception", "testing"]:
        history_context = conversation_history.get_conversation_context()
        if history_context and len(history_context) < 500:
            prompt_parts.insert(-2, f"Previous: {history_context[:300]}")
    final_prompt = "\n".join(prompt_parts)
    print(f"Compact prompt: {len(final_prompt)} chars (was probably >30k)")
    return final_prompt


def post_process_answer(answer: str, intent: Dict[str, Any]) -> str:
    processed_answer = answer.strip()
    if len(processed_answer) < 50:
        processed_answer += "\n\nNo detailed information found in the provided code."
    return processed_answer
