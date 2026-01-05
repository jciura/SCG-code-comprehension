from typing import Any, Dict, List

from loguru import logger

from src.core.config import GPT_MODEL, RAGAS_MAX_TOKENS, RAGAS_TIMEOUT

RAGAS_AVAILABLE = False
llm = None

try:
    from datasets import Dataset
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    llm = LangchainLLMWrapper(
        ChatOpenAI(model=GPT_MODEL, timeout=RAGAS_TIMEOUT, max_tokens=RAGAS_MAX_TOKENS)
    )
    RAGAS_AVAILABLE = True

except ImportError as e:
    logger.warning(f"RAGAS not available {e}")


class RAGMetrics:
    def __init__(self):
        """
        Initialize metrics.
        """
        self.ragas_available = RAGAS_AVAILABLE

    def full_evaluation(
        self,
        question: str,
        answer: str,
        retrieved_contexts: List[str],
        ground_truth_contexts: List[str],
        key_entities: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Full evaluation.

        Args:
            question: Original question
            answer: Generated answer
            retrieved_contexts: Contexts retrieved by RAG
            ground_truth_contexts: Expected contexts
            key_entities: Key entities to check

        Returns:
            Dict with "ragas" scores and "optional_metrics"
        """
        entity_recall = 0.0
        if key_entities:
            context_text = " ".join(retrieved_contexts).lower()
            found_entities = 0
            for entity in key_entities:
                if entity.lower() in context_text:
                    found_entities += 1
            entity_recall = found_entities / len(key_entities)
        if not self.ragas_available:
            return {
                "ragas": {"error": "RAGAS not available"},
                "optional_metrics": {"context_entity_recall": round(entity_recall, 3)},
            }
        if not answer or not answer.strip():
            return {
                "ragas": {"error": "Empty answer"},
                "optional_metrics": {"context_entity_recall": round(entity_recall, 3)},
            }
        if not ground_truth_contexts:
            return {
                "ragas": {"error": "Missing ground_truth_contexts"},
                "optional_metrics": {"context_entity_recall": round(entity_recall, 3)},
            }
        ground_truth_str = " ".join(ground_truth_contexts)
        try:
            dataset = Dataset.from_dict(
                {
                    "question": [question],
                    "answer": [answer],
                    "contexts": [retrieved_contexts],
                    "ground_truth": [ground_truth_str],
                }
            )
            results = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
                llm=llm,
            )
            df = results.to_pandas()
            faithfulness_score = float(df["faithfulness"].iloc[0])
            answer_relevance_score = float(df["answer_relevancy"].iloc[0])
            context_recall_score = float(df["context_recall"].iloc[0])
            context_precision_score = float(df["context_precision"].iloc[0])
            scores = [
                faithfulness_score,
                answer_relevance_score,
                context_recall_score,
                context_precision_score,
            ]
            if min(scores) > 0:
                sum_scores = sum(1 / s for s in scores)
                ragas_score = len(scores) / sum_scores
            else:
                ragas_score = 0.0
            return {
                "ragas": {
                    "context_recall": round(context_recall_score, 3),
                    "context_precision": round(context_precision_score, 3),
                    "faithfulness": round(faithfulness_score, 3),
                    "answer_relevance": round(answer_relevance_score, 3),
                    "ragas_score": round(ragas_score, 3),
                },
                "optional_metrics": {"context_entity_recall": round(entity_recall, 3)},
            }
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {
                "ragas": {"error": str(e)},
                "optional_metrics": {"context_entity_recall": round(entity_recall, 3)},
            }
