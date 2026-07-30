import os
import subprocess
import tempfile
import librosa
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import BertTokenizer, BertModel

# Compatibility for numpy 2.x (librosa dependency)
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
if not hasattr(np, 'in1d'):
    np.in1d = np.isin
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int

class FeatureExtractor:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"FeatureExtractor initialized on {self.device}")
        
        # Load BERT for text embeddings
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.bert_model.eval()

    def extract_audio_features(self, audio_path):
        """Extract MFCC, Chroma, and Mel Spectrogram."""
        try:
            y, sr = librosa.load(audio_path, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1)
            
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            
            mel = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_mean = np.mean(mel, axis=1)
            
            return np.hstack([mfccs_mean, chroma_mean, mel_mean])
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return np.zeros(153)

    def extract_text_embedding(self, text):
        """Generate BERT embedding for text."""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return np.zeros(768)
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    def extract_image_features_from_clnf(self, clnf_au_file, clnf_feat_file):
        """Aggregate CLNF/AU features."""
        combined_feats = []
        try:
            if os.path.exists(clnf_au_file):
                df_au = pd.read_csv(clnf_au_file, skipinitialspace=True)
                au_data = df_au.drop(columns=['frame', 'timestamp'], errors='ignore')
                combined_feats.append(au_data.mean().values)
            
            if os.path.exists(clnf_feat_file):
                df_feat = pd.read_csv(clnf_feat_file, skipinitialspace=True)
                feat_data = df_feat.drop(columns=['frame', 'timestamp'], errors='ignore')
                combined_feats.append(feat_data.mean().values)
            
            if combined_feats:
                return np.concatenate(combined_feats)
        except Exception as e:
            print(f"Error extracting image features: {e}")
            
        return np.zeros(160) # Default size based on training

# Standalone helper functions for webapp compatibility
def extract_audio_features_standard(path):
    fe = FeatureExtractor()
    return fe.extract_audio_features(path)

def extract_image_features_openface(path, expected_dim=160):
    """Return the same CLNF/OpenFace feature type used during training.

    There is intentionally no ResNet fallback: same length is not the same
    feature representation. The configured extractor must produce the legacy
    CLNF AU/features files consumed by ``preprocess_images.py``.
    """
    from webapp.config import OPENFACE_IMAGE_BINARY, OPENFACE_VIDEO_BINARY

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    is_image = Path(path).suffix.lower() in image_suffixes
    binary = OPENFACE_IMAGE_BINARY if is_image else OPENFACE_VIDEO_BINARY
    env_name = "OPENFACE_IMAGE_BINARY" if is_image else "OPENFACE_VIDEO_BINARY"

    if not binary:
        raise RuntimeError(f"OpenFace is not configured. Set {env_name} to the extractor path.")
    if not os.path.isfile(binary):
        raise RuntimeError(f"OpenFace executable not found: {binary}")

    with tempfile.TemporaryDirectory(prefix="openface_") as output_dir:
        result = subprocess.run(
            [binary, "-f", os.path.abspath(path), "-out_dir", output_dir],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode != 0:
            details = (result.stderr or result.stdout).strip()[-800:]
            raise RuntimeError(f"OpenFace extraction failed: {details}")

        output_path = Path(output_dir)
        au_files = list(output_path.rglob("*_CLNF_AUs.txt"))
        feature_files = list(output_path.rglob("*_CLNF_features.txt"))
        csv_files = list(output_path.rglob("*.csv"))
        if not au_files and not feature_files and not csv_files:
            raise RuntimeError(
                "OpenFace output is not compatible with the training schema. "
                "Configure a CLNF exporter or modern OpenFace CSV exporter."
            )

        if au_files or feature_files:
            extractor = FeatureExtractor.__new__(FeatureExtractor)
            vector = extractor.extract_image_features_from_clnf(
                str(au_files[0]) if au_files else "",
                str(feature_files[0]) if feature_files else "",
            ).astype(np.float32).flatten()
        else:
            df = pd.read_csv(csv_files[0], skipinitialspace=True)
            numeric = df.select_dtypes(include=[np.number]).drop(
                columns=["frame", "face_id", "timestamp", "confidence", "success"],
                errors="ignore",
            )
            if numeric.empty:
                raise RuntimeError("OpenFace CSV did not contain numeric feature columns.")
            vector = numeric.mean(axis=0).to_numpy(dtype=np.float32).flatten()

    if vector.size < expected_dim:
        vector = np.pad(vector, (0, expected_dim - vector.size))
    elif vector.size > expected_dim:
        vector = vector[:expected_dim]

    if vector.size != expected_dim or not np.isfinite(vector).all():
        raise RuntimeError(
            f"CLNF schema mismatch: expected {expected_dim} finite features, got {vector.size}."
        )
    return vector
