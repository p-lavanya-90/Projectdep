"""SHAP Service — GradientExplainer on the Attention Fusion model."""

import os, pickle, warnings
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from webapp.config import (X_TRAIN_BERT, X_DEV_BERT, Y_DEV_BIN,
                            AUDIO_DIM, IMAGE_DIM, TEXT_DIM, MODEL_ATTN, MODEL_CMT)
from webapp.architectures import AttentionFusionModel, CrossModalTransformer

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
DEVICE = torch.device("cpu")

_shap_cache = {}


class _ConcatWrapper(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.model = base
    def forward(self, x):
        a = x[:, :AUDIO_DIM]
        i = x[:, AUDIO_DIM:AUDIO_DIM+IMAGE_DIM]
        t = x[:, AUDIO_DIM+IMAGE_DIM:]
        out, _ = self.model(a, i, t)
        return out


def _load_wrapped():
    if os.path.exists(MODEL_ATTN):
        base = AttentionFusionModel()
        base.load_state_dict(torch.load(MODEL_ATTN, map_location="cpu"))
    else:
        base = CrossModalTransformer()
        base.load_state_dict(torch.load(MODEL_CMT, map_location="cpu"))
    base.eval()
    w = _ConcatWrapper(base); w.eval()
    return w


def get_shap_explanation(n_bg: int = 30, n_explain: int = 10) -> dict:
    """
    Compute SHAP values for the dev set and return structured data
    ready for the frontend charts.
    """
    key = f"shap_{n_bg}_{n_explain}"
    if key in _shap_cache:
        return _shap_cache[key]

    try:
        import shap
    except ImportError:
        return {"error": "shap not installed — run: pip install shap"}

    X_train = np.load(X_TRAIN_BERT).astype(np.float32)
    X_dev   = np.load(X_DEV_BERT).astype(np.float32)
    y_dev   = np.load(Y_DEV_BIN)

    wrapped = _load_wrapped()

    X_bg = torch.tensor(X_train[:n_bg])
    X_ex = torch.tensor(X_dev[:n_explain])

    explainer   = shap.GradientExplainer(wrapped, X_bg)
    shap_values = explainer.shap_values(X_ex)

    # Normalise to (N, F) for class 1 (Depressed)
    if isinstance(shap_values, list):
        sv = np.array(shap_values[1])
        if sv.shape[0] != n_explain:
            sv = sv.T
    else:
        sv = np.array(shap_values)
        if sv.ndim == 3:
            sv = sv[:,:,1]
        if sv.shape[0] != n_explain:
            sv = sv.T

    # Modality-level aggregated SHAP
    sv_audio = np.abs(sv[:, :AUDIO_DIM]).mean(axis=1)
    sv_image = np.abs(sv[:, AUDIO_DIM:AUDIO_DIM+IMAGE_DIM]).mean(axis=1)
    sv_text  = np.abs(sv[:, AUDIO_DIM+IMAGE_DIM:]).mean(axis=1)

    avg_a = float(sv_audio.mean())
    avg_i = float(sv_image.mean())
    avg_t = float(sv_text.mean())
    total = avg_a + avg_i + avg_t + 1e-9

    # Top-20 BERT features
    sv_text_block = sv[:, AUDIO_DIM+IMAGE_DIM:]
    mean_bert     = np.abs(sv_text_block).mean(axis=0)
    top20_idx     = np.argsort(mean_bert)[::-1][:20]

    # Per-sample breakdown
    dep_idx = int(np.where(y_dev[:n_explain]==1)[0][0]) if (y_dev[:n_explain]==1).any() else 0

    result = {
        "modality_shap": {
            "Audio": round(avg_a, 6),
            "Image": round(avg_i, 6),
            "Text":  round(avg_t, 6),
        },
        "modality_pct": {
            "Audio": round(avg_a/total*100, 1),
            "Image": round(avg_i/total*100, 1),
            "Text":  round(avg_t/total*100, 1),
        },
        "per_sample": {
            "audio": [round(float(x),6) for x in sv_audio.tolist()],
            "image": [round(float(x),6) for x in sv_image.tolist()],
            "text":  [round(float(x),6) for x in sv_text.tolist()],
            "labels": y_dev[:n_explain].tolist(),
        },
        "top20_bert": {
            "indices": top20_idx.tolist(),
            "values":  [round(float(mean_bert[i]),6) for i in top20_idx],
            "labels":  [f"BERT-dim-{i}" for i in top20_idx],
        },
        "waterfall_sample": {
            "index":  dep_idx,
            "audio":  round(float(sv_audio[dep_idx]),6),
            "image":  round(float(sv_image[dep_idx]),6),
            "text":   round(float(sv_text[dep_idx]),6),
            "label":  "Depressed" if y_dev[dep_idx]==1 else "Non-Depressed",
        }
    }
    _shap_cache[key] = result
    return result
