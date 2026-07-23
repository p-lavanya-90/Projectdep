import os
import librosa
import numpy as np
from pathlib import Path

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

def preprocess_audio():
    base_dir = Path("c:/Users/Sampath Kumar/Downloads/project")
    raw_audio_dir = base_dir / "raw_audio"
    target_dir = base_dir / "preprocessed_audio"
    target_dir.mkdir(parents=True, exist_ok=True)

    print("Starting audio preprocessing...")

    for participant_dir in raw_audio_dir.iterdir():
        if participant_dir.is_dir():
            participant_id = participant_dir.name
            audio_files = list(participant_dir.glob("*_AUDIO.wav"))
            
            if not audio_files:
                continue
            
            audio_file = audio_files[0]
            print(f"Preprocessing Audio: {participant_id}")
            
            try:
                # Load audio
                y, sr = librosa.load(audio_file, sr=None)
                
                # Extract MFCC
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfccs_mean = np.mean(mfccs, axis=1)
                
                # Extract Chroma
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                chroma_mean = np.mean(chroma, axis=1)
                
                # Extract Mel Spectrogram
                mel = librosa.feature.melspectrogram(y=y, sr=sr)
                mel_mean = np.mean(mel, axis=1)
                
                # Combine features
                combined_features = np.hstack([mfccs_mean, chroma_mean, mel_mean])
                
                # Save
                np.save(target_dir / f"{participant_id}_audio_features.npy", combined_features)
            except Exception as e:
                print(f"Error processing {participant_id}: {e}")

    print("Audio preprocessing complete!")

if __name__ == "__main__":
    preprocess_audio()
