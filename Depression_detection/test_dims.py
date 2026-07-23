import joblib
import os
import sys

# Add the project dir to path for imports if needed
sys.path.append(r'c:\Users\Sampath Kumar\Downloads\project\Depression_detection')

MODELS_DIR     = r'c:\Users\Sampath Kumar\Downloads\project\Depression_detection\models'
FEATURE_SCALER  = os.path.join(MODELS_DIR, "feature_scaler.pkl")

try:
    if os.path.exists(FEATURE_SCALER):
        scaler = joblib.load(FEATURE_SCALER)
        if hasattr(scaler, 'n_features_in_'):
            print(f"Scaler expected features: {scaler.n_features_in_}")
        else:
            # Maybe try to check mean_ or similar
            print(f"Scaler features: {len(scaler.mean_)}")
    else:
        print(f"Scaler not found: {FEATURE_SCALER}")
except Exception as e:
    print(f"Error checking dimensions: {e}")
