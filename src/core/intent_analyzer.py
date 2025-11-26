import json
import re
from typing import Any, Dict, Optional

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.core.config import default_classifier_embeddings_path, default_classifier_model
from src.core.models import IntentAnalysis, IntentCategory


class IntentAnalyzer:
    def __init__(
        self,
        classifier_embeddings_path: str = default_classifier_embeddings_path,
        classifier_model: str = default_classifier_model,
    ):
        """
        Initializes the analyzer with paths and model identifiers.

        Args:
            classifier_embeddings_path (str): Path to JSON file with example embeddings.
            classifier_model (str): SentenceTransformer model name used for encoding.
        """
        self.classifier_embeddings_path = classifier_embeddings_path
        self.classifier_model_name = classifier_model
        self._classifier_model: Optional[SentenceTransformer] = None
        self._classifier_embeddings: Optional[Dict] = None

        self.intent_patterns = {
            IntentCategory.USAGE: {
                "keywords": [
                    "used",
                    "called",
                    "invoked",
                    "where",
                    "references",
                    "usage",
                    "utilized",
                ],
                "patterns": [r"where.*used", r"how.*called", r".*usage.*", r".*referenced.*"],
                "weight": 2.0,
            },
            IntentCategory.DEFINITION: {
                "keywords": ["what is", "describe", "explain", "define", "meaning", "purpose"],
                "patterns": [r"what\s+is", r"describe.*", r"explain.*", r"define.*"],
                "weight": 1.8,
            },
            IntentCategory.IMPLEMENTATION: {
                "keywords": [
                    "how does",
                    "implementation",
                    "algorithm",
                    "logic",
                    "works",
                    "internally",
                ],
                "patterns": [
                    r"how\s+does.*work",
                    r"implementation.*",
                    r"algorithm.*",
                    r".*logic.*",
                ],
                "weight": 1.6,
            },
            IntentCategory.TESTING: {
                "keywords": ["test", "testing", "junit", "mock", "verify", "unit test"],
                "patterns": [r".*test.*", r"junit.*", r"mock.*", r".*testing.*"],
                "weight": 1.5,
            },
            IntentCategory.EXCEPTION: {
                "keywords": ["error", "exception", "throw", "catch", "fail", "handling"],
                "patterns": [r".*error.*", r".*exception.*", r".*fail.*", r".*throw.*"],
                "weight": 1.4,
            },
            IntentCategory.TOP: {
                "keywords": [
                    "important classes",
                    "main classes",
                    "core classes",
                    "key classes",
                    "most connected",
                    "biggest class",
                    "largest class",
                    "most lines",
                    "central",
                    "core",
                    "dominant",
                    "important modules",
                    "useless",
                    "not importantnajważniejsze klasy",
                    "główne klasy",
                    "centralne klasy",
                    "klasy z największym kodem",
                    "największe klasy",
                    "najbardziej połączone klasy",
                    "najmniej ważne klasy",
                    "najmniej linii",
                    "najmniej połączeń",
                    "least",
                    "smallest classes",
                    "least lines",
                    "not core",
                    "najmniejszą",
                    "najmniejsze",
                    "najmniej",
                    "min",
                    "max",
                ],
                "patterns": [
                    r"najważniejsz.*klas",
                    r"główn.*klas",
                    r"centraln.*klas",
                    r"największ.*klas",
                    r"połączon.*klas",
                    r"klas.*największ.*kod",
                    r"most\s+(connected|important|significant|central|biggest|dominant).*class",
                    r"key\s+class",
                    r"core\s+class",
                    r"main\s+class",
                    r"najmniejsz.*klas",
                ],
                "weight": 1.2,
            },
        }

        self.context_limits = {
            IntentCategory.DEFINITION: {
                "max_tokens": 6000,
                "max_context_chars": 36000,
                "base_nodes": 4,
                "category_nodes": 3,
                "fill_nodes": 2,
            },
            IntentCategory.IMPLEMENTATION: {
                "max_tokens": 7200,
                "max_context_chars": 30000,
                "base_nodes": 5,
                "category_nodes": 4,
                "fill_nodes": 3,
            },
            IntentCategory.USAGE: {
                "max_tokens": 4800,
                "max_context_chars": 24000,
                "base_nodes": 3,
                "category_nodes": 4,
                "fill_nodes": 2,
            },
            IntentCategory.TESTING: {
                "max_tokens": 4800,
                "max_context_chars": 30000,
                "base_nodes": 4,
                "category_nodes": 4,
                "fill_nodes": 3,
            },
            IntentCategory.EXCEPTION: {
                "max_tokens": 6000,
                "max_context_chars": 28000,
                "base_nodes": 4,
                "category_nodes": 3,
                "fill_nodes": 2,
            },
            IntentCategory.GENERAL: {
                "max_tokens": 4000,
                "max_context_chars": 16000,
                "base_nodes": 3,
                "category_nodes": 2,
                "fill_nodes": 2,
            },
        }

    @property
    def classifier_model(self) -> SentenceTransformer:
        """
        Lazily loads and returns the sentence-transformer classifier model.

        Returns:
            SentenceTransformer: Loaded model instance.
        """
        if self._classifier_model is None:
            self._classifier_model = SentenceTransformer(self.classifier_model_name)
        return self._classifier_model

    @property
    def classifier_embeddings(self) -> Dict:
        """
        Lazily loads and returns classifier example embeddings from disk.

        Returns:
            Dict: Mapping of label -> list of embedding vectors.

        Raises:
            FileNotFoundError: If the embeddings file cannot be found.
        """
        if self._classifier_embeddings is None:
            try:
                with open(self.classifier_embeddings_path, "r", encoding="utf-8") as f:
                    self._classifier_embeddings = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Classifier embeddings not found at {self.classifier_embeddings_path}"
                )
        return self._classifier_embeddings

    def classify_question_basic(self, question: str) -> str:
        """
        Classifies a question using cosine similarity against example embeddings.

        Args:
            question (str): Natural-language question text.

        Returns:
            str: Best-matching label (string value of `IntentCategory`).
        """
        question_emb = self.classifier_model.encode([question], convert_to_tensor=False)[0]
        best_score = -1
        best_label = IntentCategory.GENERAL.value
        for label, examples in self.classifier_embeddings.items():
            for emb in examples:
                score = cosine_similarity([question_emb], [emb])[0][0]
                if score > best_score:
                    best_score = score
                    best_label = label
        return best_label

    def enhanced_classify_question(self, question: str) -> IntentAnalysis:
        """
        Produces an enhanced intent analysis with heuristics and confidence.

        Starts from the basic embedding-based label, maps legacy labels, then
        applies keyword and regex pattern scores to derive a primary intent,
        confidence, secondary intents, and requirement flags.

        Args:
            question (str): Question text to analyze.

        Returns:
            IntentAnalysis: Structured analysis with primary intent and details.
        """
        basic_category = self.classify_question_basic(question)

        category_mapping = {"medium": "definition", "specific": "implementation"}
        if basic_category in category_mapping:
            basic_category = category_mapping[basic_category]

        question_lower = question.lower()
        scores = {}
        for category, config in self.intent_patterns.items():
            score = 0.0
            for keyword in config["keywords"]:
                if keyword in question_lower:
                    score += config["weight"]
            for pattern in config["patterns"]:
                if re.search(pattern, question_lower):
                    score += config["weight"] * 1.2
            scores[category.value] = score

        if all(score == 0.0 for score in scores.values()):
            return IntentAnalysis(
                primary_intent=IntentCategory(basic_category),
                confidence=0.5,
                scores=scores,
                requires_examples=False,
                requires_usage_info=False,
                requires_implementation_details=False,
                enhanced=False,
            )
        requires_examples = False
        requires_usage_info = False
        requires_implementation_details = False
        best_category = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_category] / total_score if total_score > 0 else 0.5
        primary_intent_enum = IntentCategory(best_category)
        if primary_intent_enum == IntentCategory.USAGE:
            requires_usage_info = True
        elif primary_intent_enum in [
            IntentCategory.DEFINITION,
            IntentCategory.TESTING,
            IntentCategory.EXCEPTION,
        ]:
            requires_examples = True
        elif primary_intent_enum == IntentCategory.IMPLEMENTATION:
            requires_implementation_details = True
        return IntentAnalysis(
            primary_intent=primary_intent_enum,
            confidence=min(confidence, 1.0),
            scores=scores,
            requires_examples=requires_examples,
            requires_usage_info=requires_usage_info,
            requires_implementation_details=requires_implementation_details,
            enhanced=True,
        )

    def get_context_limits(self, intent: IntentCategory) -> Dict[str, Any]:
        """
        Returns token/section limits for the given intent.

        Args:
            intent (IntentCategory): Target intent category.

        Returns:
            Dict[str, Any]: Limits including `max_tokens`, `max_context_chars`,
                `base_nodes`, `category_nodes`, and `fill_nodes`.
        """
        return self.context_limits.get(
            intent,
            {
                "max_tokens": 600,
                "max_context_chars": 3000,
                "base_nodes": 2,
                "category_nodes": 1,
                "fill_nodes": 1,
            },
        )

    def is_usage_question(self, question: str) -> bool:
        """
        Checks if a question likely asks about *usage*.

        Args:
            question (str): Question text.

        Returns:
            bool: True if usage-related keywords are present.
        """
        question_lower = question.lower()
        return any(
            word in question_lower for word in ["used", "where", "usage", "called", "referenced"]
        )


_intent_analyzer: Optional[IntentAnalyzer] = None


def get_intent_analyzer() -> IntentAnalyzer:
    """
    Returns a cached singleton of `IntentAnalyzer`.

    Lazily constructs the analyzer on first access.

    Returns:
        IntentAnalyzer: Shared analyzer instance.
    """
    global _intent_analyzer
    if _intent_analyzer is None:
        _intent_analyzer = IntentAnalyzer()
    return _intent_analyzer


def classify_question(question: str) -> str:
    analyzer = get_intent_analyzer()
    return analyzer.classify_question_basic(question)
