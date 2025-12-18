from typing import List, Dict, Any
from loguru import logger
from dotenv import load_dotenv

RAGAS_AVAILABLE = False
llm = None
load_dotenv()
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    RAGAS_AVAILABLE = True
    logger.info("RAGAS loaded")

except ImportError as e:
    logger.warning(f"RAGAS not fully installed: {e}")


class RAGMetrics:

    def __init__(self, embedding_model_name: str = None):
        """
        Initialize metrics.

        Args:
            embedding_model_name: Ignored
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
            found = sum(1 for e in key_entities if e.lower() in context_text)
            entity_recall = found / len(key_entities)

        if not self.ragas_available:
            return {
                "ragas": {
                    "context_recall": 0.0,
                    "faithfulness": 0.0,
                    "answer_relevance": 0.0,
                    "ragas_score": 0.0,
                    "error": "RAGAS not installed"
                },
                "optional_metrics": {
                    "context_entity_recall": round(entity_recall, 3)
                }
            }

        if not answer or not answer.strip():
            return {
                "ragas": {
                    "context_recall": 0.0,
                    "faithfulness": 0.0,
                    "answer_relevance": 0.0,
                    "ragas_score": 0.0,
                },
                "optional_metrics": {
                    "context_entity_recall": round(entity_recall, 3)
                }
            }
        if ground_truth_contexts:
            ground_truth_str = " ".join(ground_truth_contexts)
        else:
            ground_truth_str = ""
        try:
            dataset = Dataset.from_dict({
                "question": [question],
                "answer": [answer],
                "contexts": [retrieved_contexts],
                "ground_truth": [ground_truth_str] if ground_truth_str else [answer]
            })
            results = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
                llm=llm)
            df = results.to_pandas()
            faith = float(df["faithfulness"].iloc[0])
            relevance = float(df["answer_relevancy"].iloc[0])
            context_recall_lib = float(df["context_recall"].iloc[0])
            context_precision_lib = float(df["context_precision"].iloc[0])
            scores = [faith, relevance, context_recall_lib]
            if all(s > 0 for s in scores):
                sum_scores = sum(1 / s for s in scores)
                ragas_score = len(scores) / sum_scores
            else:
                ragas_score = 0.0
            return {
                "ragas": {
                    "context_recall": round(context_recall_lib, 3),
                    "context_precision": round(context_precision_lib, 3),
                    "faithfulness": round(faith, 3),
                    "answer_relevance": round(relevance, 3),
                    "ragas_score": round(ragas_score, 3),
                },
                "optional_metrics": {
                    "context_entity_recall": round(entity_recall, 3)
                }
            }

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {
                "ragas": {
                    "context_recall": 0.0,
                    "faithfulness": 0.0,
                    "answer_relevance": 0.0,
                    "ragas_score": 0.0,
                    "error": str(e)
                },
                "optional_metrics": {
                    "context_entity_recall": round(entity_recall, 3)
                }
            }