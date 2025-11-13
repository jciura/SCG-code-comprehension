from typing import Dict, Any
from loguru import logger
from src.clients.llm_client import call_llm


class LLMJudge:
    def __init__(self, model_name: str = None):
        """
            Initializes the judge with a target LLM model.

            Args:
                model_name (str, optional): Name/ID of the LLM to call. If not provided,
                    the default `MODEL_NAME` from config is used.
                """
        from src.core.config import MODEL_NAME
        self.model_name = model_name or MODEL_NAME


    async def evaluate_answer(self, question: str, generated_answer: str,
            ground_truth_answer: str, key_points: list[str],evaluation_criteria: Dict[str, str] = None) -> Dict[str, Any]:
        """
            Evaluates a generated answer against the ground truth using an LLM.

            Sends a structured prompt to the LLM asking for JSON with fields like
            `overall_score`, `factual_accuracy`, `completeness`, `criteria_compliance`,
            and `hallucination_detected`. Attempts to robustly parse JSON from the reply,
            with graceful degradation if the response is malformed.

            Args:
                question (str): The original user question.
                generated_answer (str): The model-produced answer to be evaluated.
                ground_truth_answer (str): Reference answer for comparison.
                key_points (list[str]): Checklist of key points that should be covered.
                evaluation_criteria (Dict[str, str], optional): Additional rubric items
                    (name -> description) to guide the judgment.

            Returns:
                Dict[str, Any]: Parsed evaluation dictionary. On parsing failure,
                returns a fallback structure with defaulted scores and error details.
        """
        criteria_text = ""
        if evaluation_criteria:
            criteria_text = "\n".join([f"- {k}: {v}" for k, v in evaluation_criteria.items()])

        prompt = f"""Jesteś ekspertem oceniającym jakość odpowiedzi systemów RAG.
Pytanie:
{question}

Poprawna odpowiedź (Ground Truth):
{ground_truth_answer}

Odpowiedź wygenerowana 
{generated_answer}

Kluczowe punkty do sprawdzenia:
{chr(10).join([f"{i + 1}. {point}" for i, point in enumerate(key_points)])}

{f"Kryteria: {criteria_text}" if criteria_text else ""}

**Zadanie:**
Oceń odpowiedź (0.0-1.0) pod kątem:
1. Faktycznej poprawności
2. Kompletności
3. Halucynacji
4. Zgodności z kryteriami

Zwróć JSON (bez komentarzy):
{{
    "overall_score": 0.85,
    "factual_accuracy": 0.9,
    "completeness": 0.8,
    "criteria_compliance": 0.85,
    "hallucination_detected": false,
    "reasoning": "Krótkie uzasadnienie",
    "missing_information": ["lista"],
    "incorrect_information": ["lista"]
}}"""

        try:
            response = await call_llm(prompt, model_name=self.model_name)
            import json
            import re

            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                evaluation = json.loads(json_str)
            else:
                evaluation = json.loads(response)

            logger.debug(f"LLM Judge evaluation: {evaluation.get('overall_score', 0):.2f}")
            return evaluation

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM Judge response: {e}")
            logger.debug(f"Raw response: {response[:500]}...")

            try:
                score_match = re.search(r'"overall_score":\s*([\d.]+)', response)
                if score_match:
                    score = float(score_match.group(1))
                    return {
                        "overall_score": score,
                        "factual_accuracy": score,
                        "completeness": score,
                        "criteria_compliance": score,
                        "hallucination_detected": False,
                        "reasoning": "Partial parse from malformed JSON",
                        "error": str(e)
                    }
            except:
                pass

            return {
                "overall_score": 0.0,
                "factual_accuracy": 0.0,
                "completeness": 0.0,
                "criteria_compliance": 0.0,
                "hallucination_detected": True,
                "reasoning": "Failed to parse judge response",
                "error": str(e)
            }

        except Exception as e:
            logger.error(f"LLM Judge error: {e}")
            return {
                "overall_score": 0.0,
                "factual_accuracy": 0.0,
                "completeness": 0.0,
                "criteria_compliance": 0.0,
                "hallucination_detected": True,
                "reasoning": f"Evaluation failed: {str(e)}",
                "error": str(e)
            }