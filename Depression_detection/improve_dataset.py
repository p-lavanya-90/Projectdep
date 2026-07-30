from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_SEED = 42
BASE_DIR = Path(__file__).resolve().parent
LABELS_DIR = BASE_DIR / "labels"
OUT_DIR = BASE_DIR / "improved_dataset"
MODELS_DIR = BASE_DIR / "models"

FEATURE_SPECS = {
    "audio": ("preprocessed_audio", "audio_features", 153),
    "image": ("preprocessed_images", "image_features", 160),
    "text": ("preprocessed_text", "text_features", 768),
}


@dataclass
class LoadedSplit:
    name: str
    participant_ids: list[str]
    x: np.ndarray
    y_score: np.ndarray
    y_binary: np.ndarray
    status: pd.DataFrame


def _label_columns(df: pd.DataFrame) -> tuple[str, str, str | None]:
    normalized = {c.lower(): c for c in df.columns}
    pid_col = next((original for lower, original in normalized.items() if "participant_id" in lower), None)
    score_col = next((original for lower, original in normalized.items() if "score" in lower), None)
    bin_col = next((original for lower, original in normalized.items() if "binary" in lower), None)
    if pid_col is None or score_col is None:
        raise ValueError("Label CSV must contain Participant_ID and PHQ score columns.")
    return pid_col, score_col, bin_col


def _load_feature(participant_id: str, modality: str) -> tuple[np.ndarray, str]:
    folder_name, suffix, dim = FEATURE_SPECS[modality]
    path = BASE_DIR / folder_name / f"{participant_id}_{suffix}.npy"
    if not path.exists():
        return np.full(dim, np.nan, dtype=float), "missing"

    try:
        arr = np.asarray(np.load(path), dtype=float).reshape(-1)
    except Exception:
        return np.full(dim, np.nan, dtype=float), "corrupt"

    if arr.size != dim:
        fixed = np.full(dim, np.nan, dtype=float)
        fixed[: min(dim, arr.size)] = arr[:dim]
        arr = fixed
        status = "wrong_dimension"
    elif not np.isfinite(arr).all():
        status = "non_finite"
    elif np.allclose(arr, 0.0):
        status = "all_zero"
    else:
        status = "ok"
    return arr, status


def load_split(csv_name: str, split_name: str) -> LoadedSplit:
    df = pd.read_csv(LABELS_DIR / csv_name)
    pid_col, score_col, bin_col = _label_columns(df)

    rows: list[np.ndarray] = []
    statuses: list[dict[str, object]] = []
    participant_ids: list[str] = []
    y_score: list[float] = []
    y_binary: list[int] = []

    for _, row in df.iterrows():
        participant_id = str(int(row[pid_col]))
        participant_ids.append(participant_id)
        y_score.append(float(row[score_col]))
        if bin_col and not pd.isna(row[bin_col]):
            y_binary.append(int(row[bin_col]))
        else:
            y_binary.append(int(float(row[score_col]) >= 10))

        feature_parts = []
        status_row: dict[str, object] = {
            "split": split_name,
            "participant_id": participant_id,
            "phq_score": float(row[score_col]),
            "phq_binary": y_binary[-1],
        }
        for modality in FEATURE_SPECS:
            arr, status = _load_feature(participant_id, modality)
            feature_parts.append(arr)
            status_row[f"{modality}_status"] = status
            status_row[f"{modality}_missing_or_bad"] = int(status != "ok")
        status_row["all_modalities_bad"] = int(
            all(status_row[f"{m}_missing_or_bad"] for m in FEATURE_SPECS)
        )
        statuses.append(status_row)
        rows.append(np.hstack(feature_parts))

    return LoadedSplit(
        name=split_name,
        participant_ids=participant_ids,
        x=np.vstack(rows),
        y_score=np.asarray(y_score),
        y_binary=np.asarray(y_binary),
        status=pd.DataFrame(statuses),
    )


def compute_train_imputation_values(x_train: np.ndarray) -> np.ndarray:
    means = np.nanmean(np.where(np.isfinite(x_train), x_train, np.nan), axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    return means


def apply_imputation(x: np.ndarray, train_means: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    if bad.any():
        x[bad] = np.take(train_means, np.where(bad)[1])
    return x


def random_oversample(x: np.ndarray, y_binary: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    classes, counts = np.unique(y_binary, return_counts=True)
    max_count = int(counts.max())
    x_parts = [x]
    y_parts = [y_binary]
    score_parts = [y_score]

    for cls, count in zip(classes, counts):
        if count == max_count:
            continue
        cls_indices = np.flatnonzero(y_binary == cls)
        sampled = rng.choice(cls_indices, size=max_count - int(count), replace=True)
        x_parts.append(x[sampled])
        y_parts.append(y_binary[sampled])
        score_parts.append(y_score[sampled])

    x_bal = np.vstack(x_parts)
    y_bal = np.concatenate(y_parts)
    score_bal = np.concatenate(score_parts)
    order = rng.permutation(len(y_bal))
    return x_bal[order], y_bal[order], score_bal[order]


def augment_training_data(x: np.ndarray, y_binary: np.ndarray, y_score: np.ndarray, copies: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if copies <= 0:
        return x, y_binary, y_score
    rng = np.random.default_rng(RANDOM_SEED)
    feature_std = np.std(x, axis=0)
    feature_std = np.where(feature_std > 0, feature_std, 1.0)
    augmented_x = [x]
    augmented_y = [y_binary]
    augmented_score = [y_score]
    for _ in range(copies):
        depressed_indices = np.flatnonzero(y_binary == 1)
        non_depressed_indices = np.flatnonzero(y_binary == 0)
        keep_indices = np.concatenate(
            [
                depressed_indices,
                rng.choice(non_depressed_indices, size=max(1, len(depressed_indices) // 2), replace=False),
            ]
        )
        noise = rng.normal(loc=0.0, scale=0.015, size=x[keep_indices].shape) * feature_std
        augmented_x.append(x[keep_indices] + noise)
        augmented_y.append(y_binary[keep_indices])
        augmented_score.append(y_score[keep_indices])
    return np.vstack(augmented_x), np.concatenate(augmented_y), np.concatenate(augmented_score)


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    best = (0.5, -1.0, -1.0, -1.0)
    for threshold in np.arange(0.05, 0.96, 0.01):
        pred = (probabilities >= threshold).astype(int)
        scores = (
            accuracy_score(y_true, pred),
            matthews_corrcoef(y_true, pred),
            f1_score(y_true, pred, zero_division=0),
        )
        if scores > best[1:]:
            best = (float(threshold), *scores)
    return best[0], {"Dev_Accuracy": best[1], "Dev_MCC": best[2], "Dev_F1": best[3]}


def evaluate_models(x_train: np.ndarray, y_train: np.ndarray, x_dev: np.ndarray, y_dev: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    models = {
        "Improved_LogisticRegression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_SEED),
        "Improved_RandomForest": RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=RANDOM_SEED),
        "Improved_ExtraTrees": ExtraTreesClassifier(n_estimators=700, class_weight="balanced", random_state=RANDOM_SEED),
        "Improved_GradientBoosting": GradientBoostingClassifier(n_estimators=80, learning_rate=0.07, max_depth=2, random_state=RANDOM_SEED),
        "Improved_SVC_RBF": SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=RANDOM_SEED),
    }
    models["Improved_SoftVotingEnsemble"] = VotingClassifier(
        estimators=[
            ("lr", models["Improved_LogisticRegression"]),
            ("rf", models["Improved_RandomForest"]),
            ("et", models["Improved_ExtraTrees"]),
            ("gb", models["Improved_GradientBoosting"]),
            ("svc", models["Improved_SVC_RBF"]),
        ],
        voting="soft",
        weights=[2, 1, 1, 2, 1],
    )

    rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        dev_prob = model.predict_proba(x_dev)[:, 1]
        threshold, dev_scores = select_threshold(y_dev, dev_prob)
        test_prob = model.predict_proba(x_test)[:, 1]
        pred = (test_prob >= threshold).astype(int)
        row = {
            "model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "AUC": roc_auc_score(y_test, test_prob),
            "MCC": matthews_corrcoef(y_test, pred),
            "Threshold": threshold,
            **dev_scores,
        }
        rows.append(row)
        joblib.dump(model, MODELS_DIR / f"{name}_classifier.pkl")
        print(
            f"{name:28} Acc={row['Accuracy']:.4f} Precision={row['Precision']:.4f} "
            f"Recall={row['Recall']:.4f} F1={row['F1']:.4f} AUC={row['AUC']:.4f}"
        )
    return pd.DataFrame(rows).sort_values(["Accuracy", "MCC", "F1"], ascending=False)


def write_audit_report(splits: list[LoadedSplit]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    status = pd.concat([split.status for split in splits], ignore_index=True)
    status.to_csv(OUT_DIR / "dataset_quality_audit.csv", index=False)

    summary_rows = []
    for split in splits:
        row = {"split": split.name, "records": len(split.y_binary)}
        row["non_depressed"] = int(np.sum(split.y_binary == 0))
        row["depressed"] = int(np.sum(split.y_binary == 1))
        for modality in FEATURE_SPECS:
            row[f"{modality}_bad_count"] = int(split.status[f"{modality}_missing_or_bad"].sum())
        row["all_modalities_bad_count"] = int(split.status["all_modalities_bad"].sum())
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "dataset_quality_summary.csv", index=False)


def write_leakage_report() -> None:
    rows = []
    for csv_name in ["train_split_Depression_AVEC.csv", "dev_split_Depression_AVEC.csv", "full_test_split.csv"]:
        df = pd.read_csv(LABELS_DIR / csv_name)
        pid_col, score_col, bin_col = _label_columns(df)
        suspicious = [
            c for c in df.columns
            if c not in {pid_col, score_col, bin_col}
            and any(token in c.lower() for token in ["phq", "score", "binary", "label", "depression"])
        ]
        rows.append({
            "file": csv_name,
            "participant_id_column": pid_col,
            "score_column": score_col,
            "binary_column": bin_col or "",
            "excluded_metadata_columns": ",".join(c for c in df.columns if c not in {pid_col, score_col, bin_col}),
            "suspicious_unused_columns": ",".join(suspicious),
            "leakage_status": "OK: model uses only .npy modality features and target labels",
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "label_leakage_check.csv", index=False)


def write_external_data_template() -> None:
    template = OUT_DIR / "external_training_data_template.csv"
    with template.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["participant_id", "phq_score", "phq_binary", "audio_feature_path", "image_feature_path", "text_feature_path", "source_dataset"])
        writer.writerow(["example_001", "12", "1", "path/to/audio.npy", "path/to/image.npy", "path/to/text.npy", "external_train_only"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an auditable improved dataset and run accuracy-focused evaluation.")
    parser.add_argument("--augment-copies", type=int, default=1, help="Number of train-only Gaussian augmentation passes.")
    args = parser.parse_args()

    train = load_split("train_split_Depression_AVEC.csv", "train")
    dev = load_split("dev_split_Depression_AVEC.csv", "dev")
    test = load_split("full_test_split.csv", "test")
    write_audit_report([train, dev, test])
    write_leakage_report()
    write_external_data_template()

    train_means = compute_train_imputation_values(train.x)
    x_train = apply_imputation(train.x, train_means)
    x_dev = apply_imputation(dev.x, train_means)
    x_test = apply_imputation(test.x, train_means)

    x_train_bal, y_train_bal, y_score_bal = random_oversample(x_train, train.y_binary, train.y_score)
    x_train_aug, y_train_aug, y_score_aug = augment_training_data(
        x_train_bal, y_train_bal, y_score_bal, copies=args.augment_copies
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_aug)
    x_dev_scaled = scaler.transform(x_dev)
    x_test_scaled = scaler.transform(x_test)

    OUT_DIR.mkdir(exist_ok=True)
    np.save(OUT_DIR / "X_train_improved.npy", x_train_scaled)
    np.save(OUT_DIR / "X_dev_improved.npy", x_dev_scaled)
    np.save(OUT_DIR / "X_test_improved.npy", x_test_scaled)
    np.save(OUT_DIR / "y_train_binary_improved.npy", y_train_aug)
    np.save(OUT_DIR / "y_train_score_improved.npy", y_score_aug)
    np.save(OUT_DIR / "y_dev_binary.npy", dev.y_binary)
    np.save(OUT_DIR / "y_test_binary.npy", test.y_binary)
    joblib.dump(scaler, OUT_DIR / "improved_feature_scaler.pkl")

    pd.DataFrame(
        [
            {"step": "1_clean_or_flag_noisy_samples", "status": "done", "output": "dataset_quality_audit.csv"},
            {"step": "2_improve_transcripts", "status": "prepared", "output": "No raw transcripts found; use preprocess_text.py after better transcripts are added."},
            {"step": "3_balance_classes", "status": "done", "output": f"train rows {len(train.y_binary)} -> balanced rows {len(y_train_bal)}"},
            {"step": "4_better_feature_extraction", "status": "prepared", "output": "pipeline keeps BERT/audio/OpenFace feature slots; rerun preprocessors when improved extractors are available."},
            {"step": "5_external_training_data", "status": "template_created", "output": "external_training_data_template.csv"},
            {"step": "6_train_only_augmentation", "status": "done", "output": f"balanced rows {len(y_train_bal)} -> augmented rows {len(y_train_aug)}"},
            {"step": "7_remove_label_leakage", "status": "done", "output": "label_leakage_check.csv"},
        ]
    ).to_csv(OUT_DIR / "dataset_improvement_steps.csv", index=False)

    print("Dataset improvement files saved in:", OUT_DIR)
    print("Train rows:", len(train.y_binary), "Balanced rows:", len(y_train_bal), "Augmented rows:", len(y_train_aug))
    results = evaluate_models(x_train_scaled, y_train_aug, x_dev_scaled, dev.y_binary, x_test_scaled, test.y_binary)
    results.to_csv(OUT_DIR / "improved_accuracy_model_comparison.csv", index=False)
    print("Best improved model:")
    print(results.head(1).to_string(index=False))


if __name__ == "__main__":
    main()
