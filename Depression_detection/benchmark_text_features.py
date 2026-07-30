from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
LABELS_DIR = BASE_DIR / "labels"
TEXT_DIR = BASE_DIR / "preprocessed_text"
MODELS_DIR = BASE_DIR / "models"
FEATURE_DIM = 768
RANDOM_SEED = 42


def labels(csv_name: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    df = pd.read_csv(LABELS_DIR / csv_name)
    pid_col = next(c for c in df.columns if "participant_id" in c.lower())
    score_col = next(c for c in df.columns if "score" in c.lower())
    bin_col = next((c for c in df.columns if "binary" in c.lower()), None)
    ids = [str(int(pid)) for pid in df[pid_col]]
    scores = df[score_col].to_numpy(dtype=float)
    if bin_col:
        binary = df[bin_col].to_numpy(dtype=int)
    else:
        binary = (scores >= 10).astype(int)
    return ids, scores, binary


def load_text(ids: list[str]) -> tuple[np.ndarray, int]:
    rows = []
    missing = 0
    for pid in ids:
        path = TEXT_DIR / f"{pid}_text_features.npy"
        if path.exists():
            rows.append(np.asarray(np.load(path), dtype=float).reshape(-1)[:FEATURE_DIM])
        else:
            rows.append(np.zeros(FEATURE_DIM, dtype=float))
            missing += 1
    return np.vstack(rows), missing


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, tuple[float, float, float]]:
    best = (0.5, -1.0, -1.0, -1.0)
    for threshold in np.arange(0.05, 0.96, 0.01):
        pred = (probabilities >= threshold).astype(int)
        score = (
            accuracy_score(y_true, pred),
            matthews_corrcoef(y_true, pred),
            f1_score(y_true, pred, zero_division=0),
        )
        if score > best[1:]:
            best = (float(threshold), *score)
    return best[0], best[1:]


def main() -> None:
    train_ids, _, y_train = labels("train_split_Depression_AVEC.csv")
    dev_ids, _, y_dev = labels("dev_split_Depression_AVEC.csv")
    test_ids, _, y_test = labels("full_test_split.csv")
    x_train, missing_train = load_text(train_ids)
    x_dev, missing_dev = load_text(dev_ids)
    x_test, missing_test = load_text(test_ids)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_dev = scaler.transform(x_dev)
    x_test = scaler.transform(x_test)

    models = {
        "Text_LogisticRegression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_SEED),
        "Text_RandomForest": RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", random_state=RANDOM_SEED),
        "Text_ExtraTrees": ExtraTreesClassifier(n_estimators=600, class_weight="balanced", random_state=RANDOM_SEED),
        "Text_GradientBoosting": GradientBoostingClassifier(n_estimators=60, learning_rate=0.08, max_depth=2, random_state=RANDOM_SEED),
    }

    rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        threshold, dev_scores = select_threshold(y_dev, model.predict_proba(x_dev)[:, 1])
        prob = model.predict_proba(x_test)[:, 1]
        pred = (prob >= threshold).astype(int)
        row = {
            "model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "AUC": roc_auc_score(y_test, prob),
            "MCC": matthews_corrcoef(y_test, pred),
            "Threshold": threshold,
            "Dev_Accuracy": dev_scores[0],
            "Dev_MCC": dev_scores[1],
            "Dev_F1": dev_scores[2],
            "Missing_Train": missing_train,
            "Missing_Dev": missing_dev,
            "Missing_Test": missing_test,
        }
        rows.append(row)
        joblib.dump(model, MODELS_DIR / f"{name}_classifier.pkl")
        print(
            f"{name:24} Acc={row['Accuracy']:.4f} Precision={row['Precision']:.4f} "
            f"Recall={row['Recall']:.4f} F1={row['F1']:.4f} AUC={row['AUC']:.4f}"
        )

    output = MODELS_DIR / "text_only_benchmark.csv"
    pd.DataFrame(rows).sort_values(["Accuracy", "MCC", "F1"], ascending=False).to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
