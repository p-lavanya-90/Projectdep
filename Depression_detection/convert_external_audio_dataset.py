from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
EXTERNAL_DIR = BASE_DIR / "external_training_data"
SOURCE_DIR = EXTERNAL_DIR / "dataset-depression"
OUTPUT_AUDIO_DIR = EXTERNAL_DIR / "preprocessed_audio"
MANIFEST = EXTERNAL_DIR / "external_audio_manifest.csv"


def apply_numpy_compat() -> None:
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid
    for name, value in {
        "float": float,
        "int": int,
        "bool": bool,
        "complex": complex,
        "object": object,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def extract_audio_features(path: Path) -> np.ndarray:
    apply_numpy_compat()
    os.environ.setdefault("NUMBA_CACHE_DIR", str(EXTERNAL_DIR / ".numba_cache"))
    import librosa

    y, sr = librosa.load(path, sr=None, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio file: {path}")

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    vector = np.hstack(
        [
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(mel, axis=1),
        ]
    ).astype(np.float32)
    if vector.size != 153:
        raise ValueError(f"Expected 153 audio features, got {vector.size}: {path}")
    if not np.isfinite(vector).all():
        raise ValueError(f"Non-finite audio features: {path}")
    return vector


def label_for_folder(folder_name: str) -> tuple[int, int]:
    lower = folder_name.lower()
    if lower.startswith("depression"):
        return 1, 10
    if lower.startswith("normal"):
        return 0, 0
    raise ValueError(f"Cannot infer label from folder: {folder_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Kaggle depression speech WAV files into project audio features.")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--limit-per-class", type=int, default=0, help="Optional cap per folder for quick tests. 0 means all.")
    args = parser.parse_args()

    if not args.source_dir.exists():
        raise SystemExit(f"Source folder not found: {args.source_dir}")

    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []

    for folder in sorted(args.source_dir.iterdir()):
        if not folder.is_dir():
            continue
        try:
            phq_binary, phq_score = label_for_folder(folder.name)
        except ValueError:
            continue
        files = sorted(folder.glob("*.wav"))
        if args.limit_per_class:
            files = files[: args.limit_per_class]
        for index, wav_path in enumerate(files):
            participant_id = f"ext_audio_{folder.name}_{index:04d}"
            output_path = OUTPUT_AUDIO_DIR / f"{participant_id}_audio_features.npy"
            try:
                features = extract_audio_features(wav_path)
                np.save(output_path, features)
                rows.append(
                    {
                        "participant_id": participant_id,
                        "phq_score": phq_score,
                        "phq_binary": phq_binary,
                        "audio_feature_path": output_path,
                        "image_feature_path": "",
                        "text_feature_path": "",
                        "source_dataset": "kaggle_depression_speech",
                        "source_file": wav_path,
                    }
                )
            except Exception as exc:
                errors.append({"source_file": wav_path, "error": str(exc)})

    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(EXTERNAL_DIR / "external_audio_conversion_errors.csv", index=False)

    print(f"Converted audio samples: {len(rows)}")
    print(f"Errors: {len(errors)}")
    print(f"Wrote manifest: {MANIFEST}")
    print(f"Wrote audio features to: {OUTPUT_AUDIO_DIR}")


if __name__ == "__main__":
    main()
