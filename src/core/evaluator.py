
from loguru import logger

from src.core.models import GroundTruthTestSuite, TestQuestion
from src.core.llm_judge import LLMJudge
from src.core.rag_metrics import RAGMetrics


class RAGEvaluator:
    def __init__(self, test_suite_path: str, use_llm_judge: bool = True):
        """
            Initializes the evaluator with a test suite and metric engines.

            Args:
                test_suite_path (str): Path to the ground-truth test suite JSON.
                use_llm_judge (bool, optional): Whether to enable the LLM judge for
                    additional scoring. Defaults to True.
        """
        self.test_suite = GroundTruthTestSuite.load_from_file(test_suite_path)
        self.rag_metrics = RAGMetrics()
        self.use_llm_judge = use_llm_judge
        if use_llm_judge:
            self.llm_judge = LLMJudge()

        logger.info(f"Loaded test suite: {self.test_suite.test_suite}")
        logger.info(f"Questions to evaluate: {len(self.test_suite.questions)}")
        logger.info(f"LLM Judge: {'Enabled' if use_llm_judge else 'Disabled'}")

