"""
ML Service — loads trained deep-learning models, runs prediction, returns
structured result with attention weights and confidence breakdown.
"""

import os, pickle, warnings
os.environ.setdefault("TRANSFORMERS_CACHE",
    os.path.join(os.path.dirname(__file__), ".hf_cache"))
os.environ.setdefault("HF_HOME",
    os.path.join(os.path.dirname(__file__), ".hf_cache"))

import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webapp.config import (MODEL_ATTN, MODEL_CMT, AUDIO_SCAL, IMAGE_SCAL,
                            AUDIO_DIM, IMAGE_DIM, TEXT_DIM, SR, N_MFCC)
from webapp.architectures import AttentionFusionModel, CrossModalTransformer

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Singleton cache ─────────────────────────────────────────────────────────
_model        = None
_arch_name    = None
_audio_scaler = None
_image_scaler = None


def get_model():
    global _model, _arch_name, _audio_scaler, _image_scaler
    if _model is not None:
        return _model, _arch_name, _audio_scaler, _image_scaler

    with open(AUDIO_SCAL, "rb") as f:
        _audio_scaler = pickle.load(f)
    with open(IMAGE_SCAL, "rb") as f:
        _image_scaler = pickle.load(f)

    if os.path.exists(MODEL_ATTN):
        m = AttentionFusionModel().to(DEVICE)
        m.load_state_dict(torch.load(MODEL_ATTN, map_location=DEVICE))
        _arch_name = "Attention Fusion"
    else:
        m = CrossModalTransformer().to(DEVICE)
        m.load_state_dict(torch.load(MODEL_CMT, map_location=DEVICE))
        _arch_name = "Cross-Modal Transformer"

    m.eval()
    _model = m
    return _model, _arch_name, _audio_scaler, _image_scaler


# ── Feature extraction ──────────────────────────────────────────────────────
def extract_audio(path: str, scaler) -> np.ndarray:
    """
    Standardised audio feature extraction for multimodal prediction.
    Uses the professional audio_utils pipeline.
    """
    from webapp.services.audio_utils import extract_mfcc_professional
    return extract_mfcc_professional(path, scaler=scaler)


def extract_image_rgb(path: str, scaler) -> np.ndarray:
    from PIL import Image
    import torchvision.transforms as T
    import torchvision.models as models
    import torch.nn as nn

    tfm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    net = nn.Sequential(*list(net.children())[:-1]); net.eval()
    img = Image.open(path).convert("RGB")
    with torch.no_grad():
        vec = net(tfm(img).unsqueeze(0)).squeeze().numpy()   # (2048,)
    f = (vec - vec.min()) / (vec.max() - vec.min() + 1e-8)
    f = np.pad(f, (0, max(0, IMAGE_DIM - len(f))))[:IMAGE_DIM]
    return scaler.transform(f.reshape(1,-1))[0].astype(np.float32)


def extract_image_npy(path: str, scaler) -> np.ndarray:
    f = np.load(path).astype(np.float32).flatten()
    f = np.pad(f, (0, max(0, IMAGE_DIM - len(f))))[:IMAGE_DIM]
    return scaler.transform(f.reshape(1,-1))[0].astype(np.float32)


# ── Singleton BERT (load once, reuse for every request) ─────────────────────
_bert_tokenizer = None
_bert_model     = None


def _get_bert():
    global _bert_tokenizer, _bert_model
    if _bert_tokenizer is None:
        from transformers import BertTokenizer, BertModel
        cache_dir = os.environ.get("TRANSFORMERS_CACHE")
        os.makedirs(cache_dir, exist_ok=True)
        _bert_tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", cache_dir=cache_dir)
        _bert_model = BertModel.from_pretrained(
            "bert-base-uncased", cache_dir=cache_dir).to(DEVICE)
        _bert_model.eval()
    return _bert_tokenizer, _bert_model


def extract_text(text: str) -> np.ndarray:
    """
    Encode text with BERT (CLS token) → (768,) float32 array.
    Falls back to TF-IDF + zero-padding when BERT is unavailable.
    """
    try:
        tok, bert = _get_bert()
        inp = tok(
            text, return_tensors="pt", truncation=True,
            padding=True, max_length=128
        ).to(DEVICE)
        with torch.no_grad():
            cls = bert(**inp).last_hidden_state[:, 0, :]
        return cls.cpu().numpy()[0].astype(np.float32)
    except Exception as bert_err:
        # ── Fallback: TF-IDF vectorizer saved during training ──────────
        warnings.warn(f"BERT failed ({bert_err}), falling back to TF-IDF.")
        return _tfidf_fallback(text)


def _tfidf_fallback(text: str) -> np.ndarray:
    """TF-IDF-based fallback encoder when BERT is unavailable.
    Output is padded/truncated to TEXT_DIM (768) to match BERT shape."""
    tfidf_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "features", "final_dataset", "tfidf_vectorizer.pkl"
    )
    if os.path.exists(tfidf_path):
        with open(tfidf_path, "rb") as f:
            vec = pickle.load(f)
        arr = vec.transform([text]).toarray()[0].astype(np.float32)
    else:
        # Last resort: simple hash bag-of-words
        words = text.lower().split()
        arr = np.zeros(TEXT_DIM, dtype=np.float32)
        for w in words:
            arr[hash(w) % TEXT_DIM] += 1.0
        if arr.max() > 0:
            arr /= arr.max()
    # Pad or truncate to TEXT_DIM
    if len(arr) < TEXT_DIM:
        arr = np.pad(arr, (0, TEXT_DIM - len(arr)))
    else:
        arr = arr[:TEXT_DIM]
    return arr.astype(np.float32)


def zero_image(scaler) -> np.ndarray:
    return scaler.transform(np.zeros((1, IMAGE_DIM)))[0].astype(np.float32)


# ── Inference ───────────────────────────────────────────────────────────────
def run_prediction(audio_feat, image_feat, text_feat) -> dict:
    model, arch_name, _, _ = get_model()
    a = torch.tensor(audio_feat).unsqueeze(0).to(DEVICE)
    i = torch.tensor(image_feat).unsqueeze(0).to(DEVICE)
    t = torch.tensor(text_feat).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, attn = model(a, i, t)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred  = int(probs.argmax())
    label = "Depressed" if pred == 1 else "Non-Depressed"
    conf  = float(probs.max())
    p_dep = float(probs[1])
    p_not = float(probs[0])

    # Attention weights
    aw = attn.cpu().numpy()
    if arch_name == "Attention Fusion":
        w = aw[0]
        attn_dict = {"Audio": round(float(w[0]),4),
                     "Image": round(float(w[1]),4),
                     "Text":  round(float(w[2]),4)}
    else:
        w = aw[0].mean(axis=0)
        attn_dict = {"Audio": round(float(w[0]),4),
                     "Image": round(float(w[1]),4),
                     "Text":  round(float(w[2]),4)}

    dom = max(attn_dict, key=attn_dict.get)
    risk = "High" if p_dep >= 0.7 else ("Moderate" if p_dep >= 0.45 else "Low")

    return {
        "prediction":       label,
        "label_code":       pred,
        "confidence":       round(conf,4),
        "prob_depressed":   round(p_dep,4),
        "prob_normal":      round(p_not,4),
        "risk_level":       risk,
        "dominant_modality": dom,
        "attention_weights": attn_dict,
        "model_used":       arch_name,
        "explanation":      _build_explanation(label, p_dep, attn_dict, dom)
    }


def _build_explanation(label, p_dep, attn, dom) -> str:
    dom_pct = round(attn[dom] * 100, 1)
    if label == "Depressed":
        return (f"The model predicted Depression with {p_dep*100:.1f}% probability. "
                f"The {dom} modality carried the most weight ({dom_pct}%), suggesting "
                f"that {_modal_hint(dom, True)}. "
                "This is a research tool — always consult a licensed clinician.")
    else:
        return (f"The model predicted Non-Depressed with {(1-p_dep)*100:.1f}% probability. "
                f"Primary signal came from {dom} ({dom_pct}%), suggesting "
                f"{_modal_hint(dom, False)}. "
                "This is a research tool — always consult a licensed clinician.")


def _modal_hint(mod, depressed) -> str:
    hints = {
        "Audio": ("acoustic cues such as flat prosody or reduced vocal energy indicate depression",
                  "vocal features appear within typical range"),
        "Image": ("facial features such as reduced expressivity or action unit patterns indicate depression",
                  "facial expression patterns appear typical"),
        "Text":  ("language patterns, word choices, or sentiment in the transcript indicate depression",
                  "transcript language appears within typical range"),
    }
    return hints[mod][0] if depressed else hints[mod][1]
