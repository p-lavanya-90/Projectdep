from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
LABELS_DIR = BASE_DIR / "labels"
EXTERNAL_DIR = BASE_DIR / "external_training_data"

FEATURES = {
    "audio": ("audio_feature_path", "preprocessed_audio", "audio_features", 153),
    "image": ("image_feature_path", "preprocessed_images", "image_features", 160),
    "text": ("text_feature_path", "preprocessed_text", "text_features", 768),
}


def validate_feature(path_value: str, expected_dim: int, field_name: str) -> Path | None:
    if pd.isna(path_value) or not str(path_value).strip():
        return None
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{field_name}: feature file not found: {path}")
    vector = np.asarray(np.load(path)).reshape(-1)
    if vector.size != expected_dim:
        raise ValueError(f"{field_name}: expected {expected_dim} values, got {vector.size}: {path}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{field_name}: contains NaN/Inf values: {path}")
    return path


def write_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "participant_id": "ext_001",
                "phq_score": 12,
                "phq_binary": 1,
                "audio_feature_path": "/absolute/path/to/ext_001_audio_features.npy",
                "image_feature_path": "/absolute/path/to/ext_001_image_features.npy",
                "text_feature_path": "/absolute/path/to/ext_001_text_features.npy",
                "source_dataset": "external_train_only",
            }
        ]
    ).to_csv(path, index=False)
    print(f"Wrote template: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add external samples to training data only.")
    parser.add_argument("--manifest", type=Path, default=EXTERNAL_DIR / "external_manifest.csv")
    parser.add_argument("--write-template", action="store_true")
    args = parser.parse_args()

    if args.write_template:
        write_template(args.manifest)
        return

    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found. Create one with: python3 add_external_training_data.py --write-template")

    manifest = pd.read_csv(args.manifest)
    required = {"participant_id", "phq_score", "phq_binary", "source_dataset"}
    missing_cols = required - set(manifest.columns)
    if missing_cols:
        raise ValueError(f"Manifest missing columns: {sorted(missing_cols)}")

    for _, row in manifest.iterrows():
        participant_id = str(row["participant_id"])
        for modality, (column, folder_name, suffix, dim) in FEATURES.items():
            source = validate_feature(row.get(column, ""), dim, column)
            if source is None:
                continue
            target_dir = EXTERNAL_DIR / folder_name
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target_dir / f"{participant_id}_{suffix}.npy")

    original_train = pd.read_csv(LABELS_DIR / "train_split_Depression_AVEC.csv")
    external_labels = pd.DataFrame(
        {
            "Participant_ID": manifest["participant_id"].astype(str),
            "PHQ8_Binary": manifest["phq_binary"].astype(int),
            "PHQ8_Score": manifest["phq_score"].astype(float),
            "Gender": -1,
            "PHQ8_NoInterest": "",
            "PHQ8_Depressed": "",
            "PHQ8_Sleep": "",
            "PHQ8_Tired": "",
            "PHQ8_Appetite": "",
            "PHQ8_Failure": "",
            "PHQ8_Concentrating": "",
            "PHQ8_Moving": "",
        }
    )
    combined = pd.concat([original_train, external_labels], ignore_index=True)
    output = EXTERNAL_DIR / "train_split_with_external.csv"
    combined.to_csv(output, index=False)

    coverage_rows = []
    for _, row in manifest.iterrows():
        participant_id = str(row["participant_id"])
        coverage = {
            "participant_id": participant_id,
            "phq_score": row["phq_score"],
            "phq_binary": int(row["phq_binary"]),
            "source_dataset": row["source_dataset"],
        }
        for modality, (_, folder_name, suffix, _) in FEATURES.items():
            coverage[f"has_{modality}"] = (EXTERNAL_DIR / folder_name / f"{participant_id}_{suffix}.npy").exists()
        coverage_rows.append(coverage)
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(EXTERNAL_DIR / "external_data_coverage.csv", index=False)

    print(f"Imported external samples: {len(manifest)}")
    print(f"Wrote combined train labels: {output}")
    print(f"Wrote coverage report: {EXTERNAL_DIR / 'external_data_coverage.csv'}")
    print("Dev/test labels were not changed.")


if __name__ == "__main__":
    main()
