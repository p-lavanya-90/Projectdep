"""Central configuration — all paths and constants in one place."""

import os

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE, "final_dataset")
MODELS_DIR = os.path.join(BASE, "models")

# ── Feature dimensions (Current Pipeline) ────────────────────
AUDIO_DIM = 153
IMAGE_DIM = 160
TEXT_DIM  = 768
TOTAL_DIM = AUDIO_DIM + IMAGE_DIM + TEXT_DIM

# ── Model Paths ──────────────────────────────────────────────
MODEL_LR_CLF    = os.path.join(MODELS_DIR, "LogisticRegression_classifier.pkl")
MODEL_RF_CLF    = os.path.join(MODELS_DIR, "RandomForestClf_classifier.pkl")
MODEL_XGB_CLF   = os.path.join(MODELS_DIR, "XGBoostClf_classifier.pkl")
MODEL_RF_REG    = os.path.join(MODELS_DIR, "RandomForestReg_regressor.pkl")
MODEL_XGB_REG   = os.path.join(MODELS_DIR, "XGBoostReg_regressor.pkl")
MODEL_LGBM_REG  = os.path.join(MODELS_DIR, "LightGBMReg_regressor.pkl")
MODEL_MLP_REG   = os.path.join(MODELS_DIR, "MLP_regressor.pth")
FEATURE_SCALER  = os.path.join(MODELS_DIR, "feature_scaler.pkl")

# ── Comparison Artifacts ─────────────────────────────────────
REG_COMP_CSV = os.path.join(MODELS_DIR, "regression_comparison.csv")
CLF_COMP_CSV = os.path.join(MODELS_DIR, "classification_comparison.csv")

# ── Dataset arrays (Precalculated for EDA) ───────────────────
# Note: These are optional if analysis_service uses prep_dataset logic
X_TRAIN_BERT = os.path.join(DATA_DIR, "X_train.npy")
X_DEV_BERT   = os.path.join(DATA_DIR, "X_dev.npy")
Y_TRAIN_SCR  = os.path.join(DATA_DIR, "y_train_score.npy")
Y_DEV_SCR    = os.path.join(DATA_DIR, "y_dev_score.npy")
Y_TRAIN_BIN  = os.path.join(DATA_DIR, "y_train_binary.npy")
Y_DEV_BIN    = os.path.join(DATA_DIR, "y_dev_binary.npy")


# ── Audio preprocessing params ─────────────────────────────────────
SR        = 16000
N_MFCC    = 40

# ── API settings ───────────────────────────────────────────────────
API_TITLE   = "Multimodal Depression Detection API"
API_VERSION = "1.0.0"
CORS_ORIGINS = ["*"]

# OpenFace executables used by the website's visual branch. Set these to the
# absolute paths of compatible extractors before launching the app.
OPENFACE_IMAGE_BINARY = os.environ.get("OPENFACE_IMAGE_BINARY", "")
OPENFACE_VIDEO_BINARY = os.environ.get("OPENFACE_VIDEO_BINARY", "")
