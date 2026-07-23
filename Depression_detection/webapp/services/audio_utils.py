"""
Audio Utilities — Professional pipeline for format conversion,
feature extraction (MFCC), and sample normalization.
FIXES: WinError 2 (ffmpeg), .webm loading, and librosa compatibility.
"""

import os
import sys
import logging
import tempfile
import numpy as np
import librosa
from pydub import AudioSegment
from webapp.config import SR, N_MFCC, AUDIO_DIM

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Windows ffmpeg Path Handling ──────────────────────────────────────────────
# Pydub requires ffmpeg. If it's not in PATH, we try to find it or warn.
# Common Windows locations for ffmpeg:
_FFMPEG_PATHS = [
    # Winget default location for Gyan.FFmpeg
    os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"),
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
    os.path.join(os.environ.get("USERPROFILE", ""), "ffmpeg", "bin")
]
# Walk subdirectories to find bin/ffmpeg.exe if needed
for base_p in _FFMPEG_PATHS:
    if os.path.isdir(base_p):
        for root, dirs, files in os.walk(base_p):
            if "ffmpeg.exe" in files:
                bin_path = root
                if bin_path not in os.environ["PATH"]:
                    os.environ["PATH"] += os.pathsep + bin_path
                    logger.info(f"Added ffmpeg to PATH from discovery: {bin_path}")
                break
        break

for p in _FFMPEG_PATHS:
    if os.path.isdir(p):
        if p not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + p
            logger.info(f"Added ffmpeg to PATH: {p}")
        break

def check_ffmpeg():
    """Returns True if ffmpeg/ffprobe are accessible."""
    from pydub.utils import which
    ffmpeg_bin = which("ffmpeg")
    ffprobe_bin = which("ffprobe")
    return bool(ffmpeg_bin and ffprobe_bin)

# ──────────────────────────────────────────────────────────────────────────────

def convert_to_wav(input_path: str) -> str:
    """
    Standardises any audio (.webm, .ogg, .mp3) to a professional WAV format:
    - 16,000Hz (SR)
    - Mono (1 channel)
    - 16-bit PCM
    """
    try:
        logger.info(f"--- PRE-CONVERSION DEBUG ---")
        logger.info(f"File: {os.path.basename(input_path)}")
        logger.info(f"Exists: {os.path.exists(input_path)}")
        logger.info(f"Size: {os.path.getsize(input_path)} bytes")
        
        if not check_ffmpeg():
            from pydub.utils import which
            logger.error(f"ffmpeg path checked: {which('ffmpeg')}")
            logger.error("ffmpeg/ffprobe NOT FOUND. Please install ffmpeg and add it to your PATH.")
            raise RuntimeError("FFmpeg not found. Cannot convert non-WAV formats (.webm/recorded audio).")

        from pydub.utils import which
        logger.info(f"Using ffmpeg at: {which('ffmpeg')}")

        # Load original
        audio = AudioSegment.from_file(input_path)
        
        # Standardise
        audio = audio.set_frame_rate(SR).set_channels(1).set_sample_width(2)
        
        # Export to temp wav
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        
        audio.export(wav_path, format="wav")
        logger.info(f"Standardised WAV created: {wav_path} (SR={SR})")
        return wav_path
        
    except Exception as e:
        logger.error(f"CONVERSION CRASH: {str(e)}")
        raise RuntimeError(f"Audio conversion failed (Check if FFmpeg is installed): {str(e)}")

def extract_mfcc_professional(path: str, scaler=None) -> np.ndarray:
    """
    Extracts high-quality spectral features (MFCC, Delta, Delta-Delta)
    into a flat 1024-dim vector as per training pipeline (combine_modalities.py).
    """
    try:
        logger.info(f"--- FEATURE EXTRACTION DEBUG ---")
        y, sr = librosa.load(path, sr=SR, mono=True)
        
        # 1. Peak Normalization to handle quiet recordings
        if len(y) > 0:
            y = librosa.util.normalize(y)
        else:
            logger.warning("Empty audio detected.")
            return np.zeros(AUDIO_DIM, dtype=np.float32)

        # 2. MFCC (40) + Delta + Delta2
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)
        features = np.vstack([mfcc, d1, d2]).astype(np.float32) # (120, T)

        # 3. Global Statistics Pooling (Stage 1)
        a_mean = np.mean(features, axis=1)  # (120,)
        a_std  = np.std(features, axis=1)   # (120,)
        vec = np.concatenate([a_mean, a_std]).astype(np.float32) # (240,)
        
        # Ensure exactly 1024 (AUDIO_DIM) to match training
        if vec.size < AUDIO_DIM:
            vec = np.pad(vec, (0, AUDIO_DIM - vec.size))
        else:
            vec = vec[:AUDIO_DIM]
            
        if scaler:
            # 5. Global StandardScaler (Final Stage)
            vec = scaler.transform(vec.reshape(1, -1))[0]
            
        logger.info(f"Feature extraction aligned with training. Final Mean: {vec.mean():.4f}")
        return vec.astype(np.float32)

    except Exception as e:
        logger.error(f"EXTRACTION CRASH: {str(e)}")
        raise RuntimeError(f"Audio feature extraction failed: {str(e)}")

def cleanup_temp_file(path: str):
    """Safely cleans up temporary files."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
            logger.info(f"Deleted temp file: {os.path.basename(path)}")
    except Exception as e:
        logger.warning(f"Cleanup failed for {path}: {e}")
