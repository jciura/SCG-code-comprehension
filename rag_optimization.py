from transformers import AutoTokenizer, AutoModel
import torch
from typing import Tuple, Optional


_codebert_model: Optional[AutoModel] = None
_codebert_tokenzier: Optional[AutoTokenizer] = None
_device: Optional[torch.device] = None


def get_codebert_model() -> Tuple[AutoModel, AutoTokenizer, torch.device]:
    global _codebert_model, _codebert_tokenzier, _device
    if _codebert_model is None:
        #print("Laduje model CodeBERT")
        _codebert_tokenzier = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        _codebert_model = AutoModel.from_pretrained("microsoft/codebert-base")
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _codebert_model = _codebert_model.to(_device)
        _codebert_model.eval()
        #print(f"Model zaladowany na {_device}")
    return _codebert_model, _codebert_tokenzier, _device


def _get_cached_model() -> bool:
    get_codebert_model()
    return True