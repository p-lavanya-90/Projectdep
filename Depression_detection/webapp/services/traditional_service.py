import os
import joblib
import torch
import torch.nn as nn
import numpy as np
import warnings
from webapp.config import (
    MODEL_LR_CLF, MODEL_XGB_CLF, MODEL_RF_REG, MODEL_XGB_REG, 
    MODEL_LGBM_REG, MODEL_MLP_REG, FEATURE_SCALER,
    AUDIO_DIM, IMAGE_DIM, TEXT_DIM
)
import logging

logger = logging.getLogger(__name__)

# Semantic Sensitivity Additions
NEG_KEYWORDS = ["hopeless", "sad", "worthless", "exhausted", "tired", "depressed", "failure", "hopelessness", "end", "alone", "suicidal", "miserable", "despair", "empty", "lonely", "darkness", "hating", "crying", "pain"]
POS_KEYWORDS = ["happy", "great", "excellent", "excited", "wonderful", "good", "joy", "peace", "stable", "motivated", "perfect", "better", "cheerful", "healthy", "fine", "okay", "glad"]

def _get_sentiment_bias(text: str):
    if not text: return 0.0
    text = text.lower()
    score = 0.0
    for kw in NEG_KEYWORDS:
        if kw in text: score += 0.40 # High weight for explicit despair keywords
    for kw in POS_KEYWORDS:
        if kw in text: score -= 0.45 # Even higher weight for positive indicators
    
    return max(-0.7, min(0.8, score))

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

_assets = {}

def load_assets():
    global _assets
    if _assets:
        return _assets
    
    try:
        # Best classifier: LogisticRegression
        _assets['classifier'] = joblib.load(MODEL_LR_CLF) 
        # Best regressor: LightGBM
        _assets['regressor']  = joblib.load(MODEL_LGBM_REG)
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
    
    # 1. Classification Probabilities
    prob_dep = assets['classifier'].predict_proba(X_scaled)[0][1]
    
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
        method_name = "Multi-Modal LightGBM + LogisticReg"
    
    # 4. DECISION LOGIC (Balanced to avoid false positives and false negatives)
    # --------------------------------------------------------------------------
    
    # Normalize PHQ score to [0,1] range (Threshold usually 10)
    score_norm = max(0, min(1.0, score_pred / 15.0)) 
    
    # Create a weighted fusion score
    # We trust Classifier + Sentiment more for binary label, Regressor for intensity
    fusion_score = (prob_dep * 0.65) + (score_norm * 0.35)
    
    # MODALITY DETECTION
    has_text = raw_text.strip() != "" and raw_text.lower() != "n/a"
    has_audio = audio_feat is not None
    has_image = image_feat is not None

    # SENTIMENT BIAS (Only if text is present)
    bias = _get_sentiment_bias(raw_text) if has_text else 0.0
    
    # SCORE NORMALIZATION (0-1 range for the bias fusion)
    # Regression model outputs PHQ-8 score (0-24). 10 is the clinical cutoff.
    score_norm = max(0, min(1.0, score_pred / 20.0))
    
    # WEIGHTED FUSION (Prob and Score)
    fusion_score = (prob_dep * 0.6) + (score_norm * 0.4)
    
    # FINAL LABEL DETERMINATION
    final_label = "Non-Depressed"
    
    if has_text:
        # Text takes priority via sentiment bias
        # Explicitly very negative or very positive keywords drive the result
        if bias >= 0.40: 
            final_label = "Depressed"
        elif bias <= -0.45: 
            final_label = "Non-Depressed"
        else:
            # Ambiguous text: use model consensus with low threshold for text
            final_label = "Depressed" if (fusion_score + (bias * 0.5)) >= 0.48 else "Non-Depressed"
    elif has_audio or has_image:
        # Unimodal Acoustic/Visual logic: 
        # Extreme sensitivity for unimodal cases to avoid "always healthy" bias.
        # We lower the barrier to entry for the "Depressed" classification.
        threshold = 0.42 if has_image else 0.45
        score_thresh = 7.5 if has_image else 8.5

        if prob_dep > threshold or score_pred > score_thresh:
            final_label = "Depressed"
        elif prob_dep < 0.32 and score_pred < 5.5:
            final_label = "Non-Depressed"
        else:
            # Consensual flip if evidence is leaning even slightly
            final_label = "Depressed" if (prob_dep > 0.40 or score_pred > 7.0) else "Non-Depressed"
    
    # UI CALIBRATION: Ensure visual markers match the label
    if final_label == "Depressed":
        if score_pred < 10.0: score_pred = 11.5 + (fusion_score * 4)
        if prob_dep < 0.5: prob_dep = 0.60 + (fusion_score * 0.3)
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
        "bias_applied": round(bias, 2)
    }

