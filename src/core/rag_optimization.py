from typing import Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from src.core.config import CODEBERT_MODEL_NAME

_codebert_model: Optional[AutoModel] = None
_codebert_tokenzier: Optional[AutoTokenizer] = None
_device: Optional[torch.device] = None


def get_codebert_model() -> Tuple[AutoModel, AutoTokenizer, torch.device]:
    """
    Loads and caches the CodeBERT model, tokenizer, and device.

    Initializes the model only once and moves it to GPU if available.
    Returns cached instances on subsequent calls.

    Returns:
        Tuple[AutoModel, AutoTokenizer, torch.device]:
            - The loaded CodeBERT model.
            - Its tokenizer.
            - The active computation device (CUDA or CPU).
    """
    global _codebert_model, _codebert_tokenzier, _device
    if _codebert_model is None:
        # print("Laduje model CodeBERT")
        _codebert_tokenzier = AutoTokenizer.from_pretrained(CODEBERT_MODEL_NAME)
        _codebert_model = AutoModel.from_pretrained(CODEBERT_MODEL_NAME)
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _codebert_model = _codebert_model.to(_device)
        _codebert_model.eval()
        # print(f"Model zaladowany na {_device}")
    return _codebert_model, _codebert_tokenzier, _device


def _get_cached_model() -> bool:
    """
    Ensures the CodeBERT model is loaded and cached.

    Calls `get_codebert_model()` to initialize the model if needed.

    Returns:
        bool: True if the model is ready.
    """
    get_codebert_model()
    return True

