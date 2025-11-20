from loguru import logger

from src.core.models import GroundTruthTestSuite
from src.core.rag_metrics import RAGMetrics


class RAGEvaluator:
    def __init__(self, test_suite_path: str):
        """
        Initializes the evaluator with a test suite and metric engines.

        Args:
            test_suite_path: Path to the ground-truth test suite JSON
        """
        self.test_suite = GroundTruthTestSuite.load_from_file(test_suite_path)
        self.rag_metrics = RAGMetrics()

        logger.info(f"Loaded test suite: {self.test_suite.test_suite}")
        logger.info(f"Questions to evaluate: {len(self.test_suite.questions)}")
