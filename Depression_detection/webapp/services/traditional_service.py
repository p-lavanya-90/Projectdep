import os
import joblib
import torch
import torch.nn as nn
import numpy as np
import warnings
from webapp.config import (
    MODEL_AUDIO_ONLY_CLF, MODEL_ELASTIC_REG, MODEL_GB_ACC_CLF, MODEL_LINEAR_REG,
    MODEL_LR_CLF, MODEL_VISUAL_ONLY_CLF, MODEL_DEPVIDMOOD_VISUAL_DISTRESS_CLF,
    MODEL_DEPVIDMOOD_VISUAL_DISTRESS_SCALER, MODEL_DEPVIDMOOD_VISUAL_DISTRESS_CNN,
    MODEL_XGB_CLF, MODEL_RF_REG, MODEL_XGB_REG, MODEL_LGBM_REG, MODEL_MLP_REG, FEATURE_SCALER,
    AUDIO_DIM, IMAGE_DIM, TEXT_DIM
)
import logging

logger = logging.getLogger(__name__)

# Semantic Sensitivity Additions
NEG_KEYWORDS = [
    "hopeless", "sad", "worthless", "exhausted", "tired", "depressed",
    "failure", "hopelessness", "end", "alone", "suicidal", "suicide",
    "miserable", "despair", "empty", "lonely", "darkness", "hating",
    "crying", "pain", "end of my life", "kill myself", "can't go on",
    "cannot go on", "no reason to live", "want to die", "just die",
    "die", "sleep forever", "not wake up", "never wake up"
]
POS_KEYWORDS = ["happy", "great", "excellent", "excited", "wonderful", "good", "joy", "peace", "stable", "motivated", "perfect", "better", "cheerful", "healthy", "fine", "okay", "glad"]

def _get_sentiment_bias(text: str):
    if not text: return 0.0
    text = text.lower()
    high_risk_phrases = (
        "suicide", "suicidal", "kill myself", "end of my life",
        "no reason to live", "want to die", "just die", "sleep forever",
        "not wake up", "never wake up", "don't want to wake up",
        "do not want to wake up"
    )
    if any(phrase in text for phrase in high_risk_phrases):
        return 0.8
    score = 0.0
    for kw in NEG_KEYWORDS:
        if kw in text: score += 0.40 # High weight for explicit despair keywords
    for kw in POS_KEYWORDS:
        if kw in text: score -= 0.45 # Even higher weight for positive indicators
    
    return max(-0.7, min(0.8, score))


def _has_high_risk_text(text: str) -> bool:
    if not text:
        return False
    text = text.lower()
    high_risk_phrases = (
        "suicide", "suicidal", "kill myself", "end of my life",
        "no reason to live", "want to die", "just die", "sleep forever",
        "not wake up", "never wake up", "don't want to wake up",
        "do not want to wake up"
    )
    return any(phrase in text for phrase in high_risk_phrases)

# Re-define MLP structure for loading
class MLPRegressorModel(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegressorModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)


class VisualDistressCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64 * 6 * 6, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

_assets = {}

HYBRID_BASE_THRESHOLD = 0.62
HYBRID_BOUNDARY_THRESHOLD = 0.53
HYBRID_MEAN_PHQ_THRESHOLD = 8.0


def _try_load(path):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        logger.warning("Could not load model artifact %s: %s", path, exc)
        return None

def load_assets():
    global _assets
    if _assets:
        return _assets
    
    try:
        classifier_path = MODEL_GB_ACC_CLF if os.path.exists(MODEL_GB_ACC_CLF) else MODEL_LR_CLF
        _assets['classifier'] = joblib.load(classifier_path)
        if os.path.exists(MODEL_AUDIO_ONLY_CLF):
            _assets['audio_only'] = joblib.load(MODEL_AUDIO_ONLY_CLF)
        if os.path.exists(MODEL_VISUAL_ONLY_CLF):
            _assets['visual_only'] = joblib.load(MODEL_VISUAL_ONLY_CLF)
        if os.path.exists(MODEL_DEPVIDMOOD_VISUAL_DISTRESS_CLF) and os.path.exists(MODEL_DEPVIDMOOD_VISUAL_DISTRESS_SCALER):
            _assets['depvidmood_visual_distress'] = joblib.load(MODEL_DEPVIDMOOD_VISUAL_DISTRESS_CLF)
            _assets['depvidmood_visual_distress_scaler'] = joblib.load(MODEL_DEPVIDMOOD_VISUAL_DISTRESS_SCALER)
        if os.path.exists(MODEL_DEPVIDMOOD_VISUAL_DISTRESS_CNN):
            checkpoint = torch.load(MODEL_DEPVIDMOOD_VISUAL_DISTRESS_CNN, map_location="cpu", weights_only=True)
            visual_cnn = VisualDistressCNN()
            visual_cnn.load_state_dict(checkpoint["model_state_dict"])
            visual_cnn.eval()
            _assets['depvidmood_visual_distress_cnn'] = visual_cnn
            _assets['depvidmood_visual_distress_cnn_threshold'] = float(checkpoint.get("threshold", 0.5))

        regressors = [
            _try_load(MODEL_RF_REG),
            _try_load(MODEL_LINEAR_REG),
            _try_load(MODEL_ELASTIC_REG),
        ]
        _assets['hybrid_regressors'] = [model for model in regressors if model is not None]

        _assets['regressor'] = (
            _try_load(MODEL_RF_REG)
            or _try_load(MODEL_LINEAR_REG)
            or _try_load(MODEL_ELASTIC_REG)
        )
        if _assets['regressor'] is None:
            raise RuntimeError("No PHQ regressor artifact could be loaded.")
        # Load MLP if exists
        input_dim = AUDIO_DIM + IMAGE_DIM + TEXT_DIM
        if os.path.exists(MODEL_MLP_REG):
            mlp = MLPRegressorModel(input_dim)
            checkpoint = torch.load(MODEL_MLP_REG, map_location='cpu', weights_only=True)
            mlp.load_state_dict(checkpoint)
            mlp.eval()
            _assets['mlp'] = mlp
        
        _assets['scaler'] = joblib.load(FEATURE_SCALER)
        logger.info("Successfully loaded ML models and scaler.")
        return _assets
    except Exception as e:
        logger.error(f"Error loading assets: {e}")
        return None


def predict_audio_only(audio_feat: np.ndarray):
    assets = load_assets()
    if not assets:
        return {"error": "Models not loaded."}
    if 'audio_only' not in assets:
        return predict_traditional(text_feat=None, audio_feat=audio_feat)

    audio_vector = np.asarray(audio_feat, dtype=np.float32).flatten()
    if audio_vector.size != AUDIO_DIM:
        raise ValueError(f"Expected {AUDIO_DIM} audio features, got {audio_vector.size}.")

    audio_bundle = assets['audio_only']
    model = audio_bundle['model']
    threshold = float(audio_bundle.get('threshold', 0.28))
    prob_dep = float(model.predict_proba(audio_vector.reshape(1, -1))[0][1])
    label = "Depressed" if prob_dep >= threshold else "Non-Depressed"

    return {
        "prediction": label,
        "prob_depressed": round(prob_dep, 4),
        "prob_normal": round(float(1 - prob_dep), 4),
        "confidence": round(float(max(prob_dep, 1 - prob_dep)), 4),
        "phq_score_estimate": None,
        "method": f"Audio-only screening ({audio_bundle.get('model_name', 'AudioOnly')})",
        "status": "success",
        "threshold": threshold,
        "note": "Audio-only screening model. Multimodal input is more reliable when text/image are available."
    }


def predict_image_only(image_feat: np.ndarray):
    assets = load_assets()
    if not assets:
        return {"error": "Models not loaded."}
    if 'visual_only' not in assets:
        return predict_traditional(text_feat=None, image_feat=image_feat)

    image_vector = np.asarray(image_feat, dtype=np.float32).flatten()
    if image_vector.size != IMAGE_DIM:
        raise ValueError(f"Expected {IMAGE_DIM} visual features, got {image_vector.size}.")

    visual_bundle = assets['visual_only']
    model = visual_bundle['model']
    threshold = float(visual_bundle.get('threshold', 0.06))
    prob_dep = float(model.predict_proba(image_vector.reshape(1, -1))[0][1])
    label = "Depressed" if prob_dep >= threshold else "Non-Depressed"

    return {
        "prediction": label,
        "prob_depressed": round(prob_dep, 4),
        "prob_normal": round(float(1 - prob_dep), 4),
        "confidence": round(float(max(prob_dep, 1 - prob_dep)), 4),
        "phq_score_estimate": None,
        "method": f"Visual-only screening ({visual_bundle.get('model_name', 'VisualOnly')})",
        "status": "success",
        "threshold": threshold,
        "note": "Visual-only screening model. A single facial image is weaker than multimodal input."
    }


def extract_depvidmood_image_features(image_path: str) -> np.ndarray:
    from PIL import Image, ImageStat

    image = Image.open(image_path).convert("L").resize((48, 48))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    hist, _ = np.histogram(arr, bins=64, range=(0.0, 1.0), density=True)
    row_mean = arr.mean(axis=1)
    col_mean = arr.mean(axis=0)
    row_std = arr.std(axis=1)
    col_std = arr.std(axis=0)
    stat = ImageStat.Stat(image)
    global_stats = np.asarray(
        [
            arr.mean(),
            arr.std(),
            arr.min(),
            arr.max(),
            np.median(arr),
            np.percentile(arr, 25),
            np.percentile(arr, 75),
            stat.rms[0] / 255.0,
        ],
        dtype=np.float32,
    )
    vector = np.concatenate([hist, row_mean, col_mean, row_std, col_std, global_stats]).astype(np.float32)
    fixed = np.zeros(IMAGE_DIM, dtype=np.float32)
    fixed[: min(IMAGE_DIM, vector.size)] = vector[:IMAGE_DIM]
    if not np.isfinite(fixed).all():
        raise ValueError("Raw image produced invalid visual distress features.")
    return fixed


def _extract_cnn_image_tensor(image_path: str):
    from PIL import Image

    image = Image.open(image_path).convert("L").resize((48, 48))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def predict_raw_image_distress(image_path: str):
    assets = load_assets()
    if not assets:
        return {"error": "Models not loaded."}

    method = "Raw image visual distress fallback"
    if 'depvidmood_visual_distress_cnn' in assets:
        with torch.no_grad():
            logits = assets['depvidmood_visual_distress_cnn'](_extract_cnn_image_tensor(image_path))
            prob_distress = float(torch.sigmoid(logits).item())
        threshold = float(assets.get('depvidmood_visual_distress_cnn_threshold', 0.5))
        method = "Raw image visual distress fallback (DepVidMood CNN expression model)"
    elif 'depvidmood_visual_distress' in assets and 'depvidmood_visual_distress_scaler' in assets:
        vector = extract_depvidmood_image_features(image_path)
        scaled = assets['depvidmood_visual_distress_scaler'].transform(vector.reshape(1, -1))
        prob_distress = float(assets['depvidmood_visual_distress'].predict_proba(scaled)[0][1])
        threshold = 0.49
        method = "Raw image visual distress fallback (DepVidMood tree expression model)"
    else:
        raise RuntimeError("DepVidMood visual distress fallback model is not available.")

    label = "Depressed" if prob_distress >= threshold else "Non-Depressed"
    score = 11.0 + (prob_distress * 5.0) if label == "Depressed" else 2.0 + (prob_distress * 5.0)

    return {
        "prediction": label,
        "prob_depressed": round(prob_distress, 4),
        "prob_normal": round(float(1 - prob_distress), 4),
        "confidence": round(float(max(prob_distress, 1 - prob_distress)), 4),
        "phq_score_estimate": round(float(score), 2),
        "method": method,
        "status": "success",
        "threshold": threshold,
        "note": (
            "Raw image was evaluated with an auxiliary expression/distress model. "
            "This improves visual-tab behavior for sad/fearful/angry expressions, "
            "but it is not a clinical PHQ-8 depression diagnosis."
        )
    }


def visual_inconclusive_result(reason: str):
    return {
        "prediction": "Inconclusive",
        "prob_depressed": 0.0,
        "prob_normal": 0.0,
        "confidence": 0.0,
        "phq_score_estimate": None,
        "method": "Visual-only expression input",
        "status": "inconclusive",
        "note": (
            "This visual upload could not be evaluated with the training-compatible "
            f"OpenFace/CLNF pipeline. {reason} Use text/audio/multimodal input or upload "
            "a 160-dimensional CLNF .npy feature file for visual-only model scoring."
        )
    }

def predict_multimodal_traditional(audio_path=None, text="", image_path=None, use_mlp=False):
    """
    Complete inference pipeline: Raw inputs -> Features -> Model.
    """
    from webapp.services.extractor_service import get_extractor
    fe = get_extractor()
    
    a_feat = fe.extract_audio_features(audio_path) if audio_path else None
    t_feat = fe.extract_text_embedding(text)
    from multimodal_utils import extract_image_features_openface
    i_feat = extract_image_features_openface(image_path, expected_dim=IMAGE_DIM) if image_path else None
    
    return predict_traditional(t_feat, a_feat, i_feat, use_mlp=use_mlp, raw_text=text)

def predict_traditional(text_feat: np.ndarray, audio_feat: np.ndarray = None, image_feat: np.ndarray = None, use_mlp=False, raw_text: str = ""):
    assets = load_assets()
    if not assets: return {"error": "Models not loaded."}
    
    # Check what we actually have (not just zeros)
    has_text = text_feat is not None and not np.all(text_feat == 0)
    has_audio = audio_feat is not None and not np.all(audio_feat == 0)
    has_image = image_feat is not None and not np.all(image_feat == 0)

    # Pad if missing
    if audio_feat is None: audio_feat = np.zeros(AUDIO_DIM)
    if image_feat is None: image_feat = np.zeros(IMAGE_DIM)
    if text_feat  is None: text_feat  = np.zeros(TEXT_DIM)
    
    X = np.hstack([audio_feat.flatten(), image_feat.flatten(), text_feat.flatten()]).reshape(1, -1)
    X_scaled = assets['scaler'].transform(X)
    
    # 1. Classification probabilities and hybrid accuracy override.
    model_prob_dep = float(assets['classifier'].predict_proba(X_scaled)[0][1])
    mean_phq_pred = None
    if assets.get('hybrid_regressors'):
        regressor_scores = [float(model.predict(X_scaled)[0]) for model in assets['hybrid_regressors']]
        mean_phq_pred = float(np.mean(regressor_scores))

    hybrid_override = (
        mean_phq_pred is not None
        and HYBRID_BOUNDARY_THRESHOLD <= model_prob_dep < HYBRID_BASE_THRESHOLD
        and mean_phq_pred >= HYBRID_MEAN_PHQ_THRESHOLD
    )
    hybrid_label = "Depressed" if model_prob_dep >= HYBRID_BASE_THRESHOLD or hybrid_override else "Non-Depressed"
    prob_dep = model_prob_dep
    
    # 2. Semantic bias for short text inputs (Very influential)
    bias = _get_sentiment_bias(raw_text)
    prob_dep = max(0.01, min(0.99, prob_dep + bias))
    
    # 3. Regression / Score prediction
    if use_mlp and 'mlp' in assets:
        with torch.no_grad():
            score_pred = assets['mlp'](torch.tensor(X_scaled, dtype=torch.float32)).item()
        method_name = "Multi-Modal Deep MLP"
    else:
        score_pred = assets['regressor'].predict(X_scaled)[0]
        method_name = "Hybrid GradientBoosting + PHQ Override"
    
    # 4. DECISION LOGIC (Balanced to avoid false positives and false negatives)
    # --------------------------------------------------------------------------
    
    # Normalize PHQ score to [0,1] range (Threshold usually 10)
    score_norm = max(0, min(1.0, score_pred / 15.0)) 
    
    # Create a weighted fusion score
    # We trust Classifier + Sentiment more for binary label, Regressor for intensity
    fusion_score = (prob_dep * 0.65) + (score_norm * 0.35)
    
    has_text = raw_text.strip() != "" and raw_text.lower() != "n/a"
    bias = _get_sentiment_bias(raw_text) if has_text else 0.0
    
    # SCORE NORMALIZATION (0-1 range for the bias fusion)
    # Regression model outputs PHQ-8 score (0-24). 10 is the clinical cutoff.
    score_norm = max(0, min(1.0, score_pred / 20.0))
    
    # WEIGHTED FUSION (Prob and Score)
    fusion_score = (prob_dep * 0.6) + (score_norm * 0.4)
    
    # FINAL LABEL DETERMINATION
    final_label = hybrid_label
    
    high_risk_text = _has_high_risk_text(raw_text)
    hf_safety = None
    if has_text:
        try:
            from webapp.services.text_safety_service import analyze_text_safety_with_hf
            hf_safety = analyze_text_safety_with_hf(raw_text)
            if hf_safety.get("risk_detected"):
                high_risk_text = True
        except Exception as exc:
            logger.warning("HF text safety check failed: %s", exc)

    if has_text and high_risk_text:
        final_label = "Depressed"
    elif has_text and bias >= 0.40:
        final_label = "Depressed"
    elif has_text and bias <= -0.45:
        final_label = "Non-Depressed"
    
    # UI CALIBRATION: Ensure visual markers match the label
    if final_label == "Depressed":
        if score_pred < 10.0: score_pred = 11.5 + (fusion_score * 4)
        if prob_dep < 0.5: prob_dep = 0.60 + (fusion_score * 0.3)
        if high_risk_text:
            score_pred = max(score_pred, 15.0)
            prob_dep = max(prob_dep, 0.88)
    else:
        if score_pred >= 10.0: score_pred = 6.2 + (fusion_score * 2)
        if prob_dep >= 0.5: prob_dep = 0.38 + (fusion_score * 0.1)

    return {
        "prediction": final_label,
        "prob_depressed": round(float(prob_dep), 4),
        "prob_normal": round(float(1 - prob_dep), 4),
        "confidence": round(float(max(prob_dep, 1-prob_dep)), 4),
        "phq_score_estimate": round(float(max(0, score_pred)), 2),
        "method": method_name,
        "status": "success",
        "bias_applied": round(bias, 2),
        "high_risk_text_detected": bool(high_risk_text),
        "hf_text_safety": hf_safety,
        "model_prob_depressed": round(model_prob_dep, 4),
        "hybrid_override_applied": bool(hybrid_override),
        "hybrid_mean_phq_score": None if mean_phq_pred is None else round(mean_phq_pred, 2)
    }
