CODE_CHARS_PER_TOKEN = 3.5
TEXT_CHARS_PER_TOKEN = 4.2


def estimate_tokens(text: str, is_code: bool = True) -> int:
    """
    Estimates token count for a text snippet.

    Uses a heuristic based on average character-per-token ratios for code or plain text.

    Args:
        text: Input text to estimate
        is_code: Whether the input is source code (default: True)

    Returns:
        Estimated token count (minimum 1)
    """
    if not text:
        return 0
    chars_per_token = CODE_CHARS_PER_TOKEN if is_code else TEXT_CHARS_PER_TOKEN
    non_whitespace_chars = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
    if len(text) > 0:
        whitespace_ratio = 1 - (non_whitespace_chars / len(text))
    else:
        whitespace_ratio = 0
    adjusted_ratio = chars_per_token * (1 + whitespace_ratio * 0.3)
    return max(1, int(len(text) / adjusted_ratio))
