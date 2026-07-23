import sys, os
import numpy as np
import torch

# Setup paths
PROJECT_ROOT = r'c:\Users\Sampath Kumar\Downloads\project\Depression_detection'
sys.path.append(PROJECT_ROOT)

from webapp.services.traditional_service import predict_traditional
from webapp.services.extractor_service import get_extractor

def test_text_prediction(sample_text="I feel very sad and hopeless."):
    print(f"Testing text prediction with: '{sample_text}'")
    try:
        fe = get_extractor()
        print("Feature extractor loaded.")
        
        t_feat = fe.extract_text_embedding(sample_text)
        print(f"Text embedding shape: {t_feat.shape}")
        
        print("Calling predict_traditional...")
        result = predict_traditional(text_feat=t_feat, raw_text=sample_text)
        print("Result:", result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_text_prediction()
