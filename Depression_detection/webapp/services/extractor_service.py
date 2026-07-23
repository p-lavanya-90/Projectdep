import torch
from multimodal_utils import FeatureExtractor

_extractor = None

def get_extractor():
    global _extractor
    if _extractor is None:
        # Force CPU to avoid CUDA initialization issues in web service on Windows
        device = torch.device("cpu")
        _extractor = FeatureExtractor(device=device)
    return _extractor
