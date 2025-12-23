import re
from typing import Any, Dict, Optional

from src.core.models import IntentAnalysis, IntentCategory


class IntentAnalyzer:
    def __init__(self):
        """
               Initialize intent patterns and context limits for each intent category.
        """
        self.intent_patterns = {
            IntentCategory.USAGE: {
                "keywords": [
                    "used", "called", "invoked", "where", "references", "usage",
                    "używana", "używane", "używany", "wywoływana", "wywoływane",
                    "wywoływany", "gdzie", "referencje", "użycie", "wykorzystanie",
                    "korzysta", "wywołuje",
                ],
                "patterns": [
                    r"where.*used", r"how.*called", r".*usage.*", r".*referenced.*",
                    r"gdzie.*używ", r"gdzie.*wywoł", r"co.*korzysta", r"kto.*używa",
                    r".*wykorzyst.*",
                ],
                "weight": 2.0,
            },
            IntentCategory.DEFINITION: {
                "keywords": [
                    "what is", "describe", "explain", "define", "meaning", "purpose",
                    "co to jest", "czym jest", "opisz", "wyjaśnij", "zdefiniuj",
                    "znaczenie", "cel", "opis", "definicja",
                ],
                "patterns": [
                    r"what\s+is", r"describe.*", r"explain.*", r"define.*",
                    r"co\s+to\s+jest", r"czym\s+jest", r"opisz.*", r"wyjaśnij.*",
                    r"zdefiniuj.*",
                ],
                "weight": 1.8,
            },
            IntentCategory.IMPLEMENTATION: {
                "keywords": [
                    "how does", "implementation", "algorithm", "logic", "works",
                    "internally", "jak działa", "implementacja", "algorytm", "logika",
                    "działanie", "wewnętrznie", "mechanizm", "sposób działania",
                ],
                "patterns": [
                    r"how\s+does.*work", r"implementation.*", r"algorithm.*",
                    r".*logic.*", r"jak\s+działa", r"jak\s+jest\s+zaimplementowan",
                    r"implementacja.*", r"algorytm.*", r".*logik.*", r"mechanizm.*",
                ],
                "weight": 1.6,
            },
            IntentCategory.TESTING: {
                "keywords": [
                    "test", "testing", "junit", "mock", "verify", "unit test",
                    "testy", "testowanie", "testowy", "mockowanie", "weryfikacja",
                    "test jednostkowy", "przypadek testowy",
                ],
                "patterns": [
                    r".*test.*", r"junit.*", r"mock.*", r".*testing.*",
                    r"jak.*testowan", r"przypadk.*testow", r".*mockow.*",
                ],
                "weight": 1.5,
            },
            IntentCategory.EXCEPTION: {
                "keywords": [
                    "error", "exception", "throw", "catch", "fail", "handling",
                    "błąd", "wyjątek", "rzucanie", "przechwytywanie",
                    "obsługa błędów", "obsługa wyjątków", "niepowodzenie",
                ],
                "patterns": [
                    r".*error.*", r".*exception.*", r".*fail.*", r".*throw.*",
                    r".*błąd.*", r".*błęd.*", r".*wyjąt.*", r".*obsług.*błęd.*",
                    r".*obsług.*wyjąt.*", r"jak.*rzuca.*wyjąt",
                ],
                "weight": 1.4,
            },
            IntentCategory.TOP: {
                "keywords": [
                    "important classes", "main classes", "core classes", "key classes",
                    "most connected", "biggest class", "largest class", "most lines",
                    "central", "core", "dominant", "important modules",
                    "najważniejsze klasy", "główne klasy", "centralne klasy",
                    "największe klasy", "najbardziej połączone klasy",
                    "najmniej ważne klasy", "least", "smallest classes",
                    "list", "show"
                ],
                "patterns": [
                    r"najważniejsz.*klas", r"główn.*klas", r"centraln.*klas",
                    r"największ.*klas", r"połączon.*klas", r"klas.*największ.*kod",
                    r"most\s+(connected|important|significant|central|biggest|dominant).*class",
                    r"key\s+class", r"core\s+class", r"main\s+class", r"najmniejsz.*klas",
                ],
                "weight": 2.0,
            },
        }
        self.context_limits = {
            IntentCategory.DEFINITION: {
                "max_chars": 500000,
                "base_nodes": 5,
                "category_nodes": 7,
                "fill_nodes": 4
            },
            IntentCategory.IMPLEMENTATION: {
                "max_chars": 500000,
                "base_nodes": 7,
                "category_nodes": 6,
                "fill_nodes": 5
            },
            IntentCategory.USAGE: {
                "max_chars": 600000,
                "base_nodes": 5,
                "category_nodes": 6,
                "fill_nodes": 4
            },
            IntentCategory.TESTING: {
                "max_chars": 400000,
                "base_nodes": 6,
                "category_nodes": 6,
                "fill_nodes": 5
            },
            IntentCategory.EXCEPTION: {
                "max_chars": 400000,
                "base_nodes": 6,
                "category_nodes": 5,
                "fill_nodes": 4
            },
            IntentCategory.GENERAL: {
                "max_chars": 600000,
                "base_nodes": 10,
                "category_nodes": 4,
                "fill_nodes": 4
            },
            IntentCategory.TOP: {
                "max_chars": 600000,
                "base_nodes": 8,
                "category_nodes": 5,
                "fill_nodes": 4
            }
        }

    def classify_question(self, question: str) -> IntentAnalysis:
        """
            Classify a user question into an intent category.

            Args:
                question (str): User question.

            Returns:
                IntentAnalysis: Detected intent, confidence and scores.
            """
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
                primary_intent=IntentCategory.GENERAL,
                confidence=0.5,
                scores=scores,
                enhanced=False,
            )
        best_category = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_category] / total_score if total_score > 0 else 0.5
        return IntentAnalysis(
            primary_intent=IntentCategory(best_category),
            confidence=min(confidence, 1.0),
            scores=scores,
            enhanced=True,
        )

    def enhanced_classify_question(self, question: str) -> IntentAnalysis:
        """
            Classify a question using enhanced analysis.

            Args:
                question (str): User question.

            Returns:
                IntentAnalysis: Intent classification result.
            """
        return self.classify_question(question)

    def classify_question_basic(self, question: str) -> str:
        """
            Classify a question and return only the intent category.

            Args:
                question (str): User question.

            Returns:
                str: Intent category value.
            """
        return self.classify_question(question).primary_intent.value

    def get_context_limits(self, intent: IntentCategory) -> Dict[str, Any]:
        """
            Return context limits for a given intent category.

            Args:
                intent (IntentCategory): Intent category.

            Returns:
                Dict[str, Any]: Context limits configuration.
            """
        return self.context_limits.get(intent, self.context_limits[IntentCategory.GENERAL])


_intent_analyzer: Optional[IntentAnalyzer] = None


def get_intent_analyzer() -> IntentAnalyzer:
    """
        Return a singleton instance of IntentAnalyzer.

        Returns:
            IntentAnalyzer: Shared analyzer instance.
        """
    global _intent_analyzer
    if _intent_analyzer is None:
        _intent_analyzer = IntentAnalyzer()
    return _intent_analyzer


def classify_question(question: str) -> str:
    """
        Classify a question and return its intent category.

        Args:
            question (str): User question.

        Returns:
            str: Intent category value.
        """
    return get_intent_analyzer().classify_question_basic(question)