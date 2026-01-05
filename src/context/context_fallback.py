import time
from typing import Any, Optional

from loguru import logger

MIN_CODE_LENGTH = 100
MAX_CONTEXT_CHARS = 40000
FALLBACK_CACHE_DURATION = 300
FALLBACK_CANDIDATES_LIMIT = 15
METHOD_MIN_LENGTH_FOR_BOOST = 250
ERROR_MESSAGE_MAX_LENGTH = 100
MAX_FALLBACK_SECTION_CHARS = 3000
FALLBACK_FETCH_LIMIT = 100

_context_cache = {"fallback": None, "timestamp": 0, "stats": {"hits": 0, "misses": 0}}


def build_fallback_context(collection: Optional[Any] = None) -> str:
    """
    Builds (and caches) a fallback context from high-importance nodes.

    When primary selection yields no sections, this function pulls top documents
    by importance from the Chroma collection and composes a compact context.

    Args:
        collection: Optional Chroma collection handle

    Returns:
        Fallback context string, or an error marker if unavailable
    """
    global _context_cache
    current_time = time.time()
    if (
            _context_cache["fallback"] is not None
            and current_time - _context_cache["timestamp"] < FALLBACK_CACHE_DURATION
    ):
        _context_cache["stats"]["hits"] += 1
        logger.debug("Using cached fallback context")
        return _context_cache["fallback"]

    _context_cache["stats"]["misses"] += 1
    if collection is None:
        try:
            from src.clients.chroma_client import get_collection

            collection = get_collection()
        except Exception as e:
            logger.error(f"Could not get collection for fallback: {e}")
            return "<NO CONTEXT AVAILABLE - COLLECTION ERROR>"

    try:
        all_nodes = collection.get(include=["metadatas", "documents"], limit=FALLBACK_FETCH_LIMIT)
        if not all_nodes["ids"]:
            return "<NO NODES AVAILABLE>"
        candidates = []
        for i in range(len(all_nodes["ids"])):
            node_id = all_nodes["ids"][i]
            doc = all_nodes["documents"][i]
            meta = all_nodes["metadatas"][i]

            if not doc or doc.startswith("<") or len(doc) < MIN_CODE_LENGTH:
                continue
            importance = float(meta.get("combined", 0.0))
            kind = meta.get("kind", "")
            if kind in ["CLASS", "TRAIT", "OBJECT"]:
                importance += 2.0
            elif kind == "INTERFACE":
                importance += 1.5
            elif kind == "METHOD" and len(doc) > METHOD_MIN_LENGTH_FOR_BOOST:
                importance += 1.0
            elif kind == "CONSTRUCTOR":
                importance += 0.5

            candidates.append((importance, node_id, doc, meta))

        if not candidates:
            return "<NO VALID CANDIDATES FOUND>"

        candidates.sort(key=lambda x: x[0], reverse=True)

        context_parts = []
        current_chars = 0
        max_fallback_chars = MAX_CONTEXT_CHARS // 2

        for importance, node_id, doc, meta in candidates[:FALLBACK_CANDIDATES_LIMIT]:
            kind = meta.get("kind", "CODE")
            doc_trimmed = doc.strip()
            if len(doc_trimmed) > MAX_FALLBACK_SECTION_CHARS:
                doc_trimmed = doc_trimmed[:MAX_FALLBACK_SECTION_CHARS] + "..."
            section = f"## {kind}: {node_id}\n{doc_trimmed}"
            section_chars = len(section)
            if current_chars + section_chars <= max_fallback_chars:
                context_parts.append(section)
                current_chars += section_chars
            else:
                break

        result = "\n\n".join(context_parts) if context_parts else "<NO HIGH-IMPORTANCE NODES FOUND>"

        _context_cache["fallback"] = result
        _context_cache["timestamp"] = current_time

        logger.debug(
            f"Built fallback context: {current_chars} chars, {len(context_parts)} sections"
        )
        return result

    except Exception as e:
        logger.error(f"Error building fallback context: {e}")
        error_context = f"<CONTEXT BUILD ERROR: {str(e)[:ERROR_MESSAGE_MAX_LENGTH]}>"
        _context_cache["fallback"] = error_context
        _context_cache["timestamp"] = current_time
        return error_context