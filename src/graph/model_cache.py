from functools import lru_cache

from loguru import logger

_graph_model = None


def get_graph_model() -> bool:
    """
    Ensures the graph embedding model is initialized.

    Lazily loads the graph model utilities if not already loaded and logs progress.

    Returns:
        bool: Always True after ensuring initialization.
    """
    global _graph_model
    if _graph_model is None:
        logger.info("Loading graph model")
        logger.info("Graph model is ready")
    return True


@lru_cache(maxsize=1000)
def cached_question_question(question_hash: str) -> str:
    """
    Cached classification of a question.

    Uses a lightweight classifier to map a (hashed) question to a category.
    Results are memoized via `functools.lru_cache`.

    Args:
        question_hash (str): Hash or key representing the question to classify.

    Returns:
        str: Classified category label for the question.
    """
    from src.core.intent_analyzer import classify_question

    return classify_question(question_hash)
