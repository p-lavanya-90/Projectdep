"""Build enhanced offline audio vectors from existing AVEC acoustic features.

Output layout per participant:
  153 baseline MFCC/chroma/Mel values
  148 COVAREP values (mean and standard deviation for 74 columns)
   10 formant values (mean and standard deviation for 5 columns)
  ---
  311 total values

This is an offline experiment. Do not use these vectors in the website until
the same COVAREP/Formant extraction is available at inference time.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
RAW = BASE / "raw_features"
BASELINE_AUDIO = BASE / "preprocessed_audio"
OUTPUT = BASE / "preprocessed_audio_enhanced"
EXPECTED_DIM = 311


def pooled_statistics(path, expected_columns):
    data = np.loadtxt(path, delimiter=",")
    data = np.atleast_2d(data).astype(np.float32)
    if data.shape[1] != expected_columns:
        raise ValueError(f"{path.name}: expected {expected_columns} columns, got {data.shape[1]}")
    return np.concatenate([data.mean(axis=0), data.std(axis=0)]).astype(np.float32)


def main():
    OUTPUT.mkdir(exist_ok=True)
    participant_ids = sorted(
        {file.stem.split("_")[0] for file in BASELINE_AUDIO.glob("*_audio_features.npy")}
        | {folder.name for folder in RAW.iterdir() if folder.is_dir()},
        key=int,
    )
    status = []
    for participant_id in participant_ids:
        baseline_path = BASELINE_AUDIO / f"{participant_id}_audio_features.npy"
        covarep_path = RAW / participant_id / f"{participant_id}_COVAREP.csv"
        formant_path = RAW / participant_id / f"{participant_id}_FORMANT.csv"

        baseline = np.load(baseline_path).astype(np.float32).flatten() if baseline_path.exists() else np.zeros(153, dtype=np.float32)
        if baseline.size != 153:
            raise ValueError(f"{baseline_path.name}: expected 153 values, got {baseline.size}")
        covarep = pooled_statistics(covarep_path, 74) if covarep_path.exists() else np.zeros(148, dtype=np.float32)
        formant = pooled_statistics(formant_path, 5) if formant_path.exists() else np.zeros(10, dtype=np.float32)
        features = np.concatenate([baseline, covarep, formant]).astype(np.float32)
        if features.size != EXPECTED_DIM or not np.isfinite(features).all():
            raise ValueError(f"{participant_id}: invalid enhanced feature vector")
        np.save(OUTPUT / f"{participant_id}_audio_features.npy", features)
        status.append({
            "participant_id": participant_id,
            "has_baseline_audio": baseline_path.exists(),
            "has_covarep": covarep_path.exists(),
            "has_formant": formant_path.exists(),
        })

    pd.DataFrame(status).to_csv(OUTPUT / "coverage.csv", index=False)
    print(f"Created {len(status)} enhanced {EXPECTED_DIM}-dimensional audio vectors in {OUTPUT}")


if __name__ == "__main__":
    main()
