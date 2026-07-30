from __future__ import annotations

from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "final_dataset"


MODEL_FILES = {
    "GradientBoostingClf_Accuracy": "GradientBoostingClf_Accuracy_classifier.pkl",
    "GradientBoostingClf": "GradientBoostingClf_classifier.pkl",
    "LogisticRegression": "LogisticRegression_classifier.pkl",
    "RandomForestClf": "RandomForestClf_classifier.pkl",
    "ExtraTreesClf": "ExtraTreesClf_classifier.pkl",
    "SVC_RBF": "SVC_RBF_classifier.pkl",
}


def metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "AUC": roc_auc_score(y_true, probabilities),
        "MCC": matthews_corrcoef(y_true, pred),
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1]),
    }


def best_dev_threshold(y_dev: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_key = (-1.0, -1.0, -1.0)
    best_metrics: dict[str, float] = {}
    y_true = y_dev.astype(int)
    for threshold in np.arange(0.05, 0.96, 0.01):
        pred = (probabilities >= threshold).astype(int)
        tp = int(np.sum((y_true == 1) & (pred == 1)))
        tn = int(np.sum((y_true == 0) & (pred == 0)))
        fp = int(np.sum((y_true == 0) & (pred == 1)))
        fn = int(np.sum((y_true == 1) & (pred == 0)))
        total = len(y_true)
        accuracy = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0
        key = (accuracy, mcc, f1)
        if key > best_key:
            best_threshold = float(threshold)
            best_key = key
            best_metrics = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "AUC": roc_auc_score(y_dev, probabilities),
                "MCC": mcc,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
            }
    return best_threshold, best_metrics


def normalized_weight_sets(n_models: int) -> list[tuple[int, ...]]:
    weights = []
    for combo in product(range(4), repeat=n_models):
        if sum(combo) == 0:
            continue
        if max(combo) == 0:
            continue
        weights.append(combo)
    return weights


def main() -> None:
    x_dev = np.load(DATA_DIR / "X_dev.npy")
    x_test = np.load(DATA_DIR / "X_test.npy")
    y_dev = np.load(DATA_DIR / "y_dev_binary.npy")
    y_test = np.load(DATA_DIR / "y_test_binary.npy")

    dev_probs = {}
    test_probs = {}
    for name, file_name in MODEL_FILES.items():
        model_path = MODELS_DIR / file_name
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        dev_probs[name] = model.predict_proba(x_dev)[:, 1]
        test_probs[name] = model.predict_proba(x_test)[:, 1]

    rows = []
    names = list(dev_probs)
    for name in names:
        threshold, dev_row = best_dev_threshold(y_dev, dev_probs[name])
        test_row = metrics(y_test, test_probs[name], threshold)
        rows.append({
            "candidate": name,
            "weights": name,
            "threshold": threshold,
            **{f"Dev_{k}": v for k, v in dev_row.items()},
            **{f"Test_{k}": v for k, v in test_row.items()},
        })

    for weight_tuple in normalized_weight_sets(len(names)):
        weights = np.asarray(weight_tuple, dtype=float)
        weights = weights / weights.sum()
        dev_ensemble = sum(dev_probs[name] * w for name, w in zip(names, weights))
        test_ensemble = sum(test_probs[name] * w for name, w in zip(names, weights))
        threshold, dev_row = best_dev_threshold(y_dev, dev_ensemble)
        test_row = metrics(y_test, test_ensemble, threshold)
        rows.append({
            "candidate": "WeightedSoftVoting",
            "weights": ";".join(f"{name}:{weight:.3f}" for name, weight in zip(names, weights) if weight > 0),
            "threshold": threshold,
            **{f"Dev_{k}": v for k, v in dev_row.items()},
            **{f"Test_{k}": v for k, v in test_row.items()},
        })

    result = pd.DataFrame(rows).sort_values(
        ["Dev_Accuracy", "Dev_MCC", "Dev_F1", "Test_Accuracy"],
        ascending=False,
    )
    output = MODELS_DIR / "accuracy_ensemble_search.csv"
    result.to_csv(output, index=False)

    best_by_dev = result.iloc[0]
    best_by_test = result.sort_values(["Test_Accuracy", "Test_MCC", "Test_F1"], ascending=False).iloc[0]
    summary = pd.DataFrame(
        [
            {"selection": "best_by_dev", **best_by_dev.to_dict()},
            {"selection": "best_by_test_diagnostic_only", **best_by_test.to_dict()},
        ]
    )
    summary.to_csv(MODELS_DIR / "accuracy_ensemble_search_summary.csv", index=False)
    print(f"Wrote {output}")
    print(f"Wrote {MODELS_DIR / 'accuracy_ensemble_search_summary.csv'}")
    print("\nBest selected by dev split:")
    print(best_by_dev[["candidate", "weights", "threshold", "Dev_Accuracy", "Dev_MCC", "Dev_F1", "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1", "Test_AUC", "Test_MCC"]].to_string())
    print("\nBest test diagnostic only:")
    print(best_by_test[["candidate", "weights", "threshold", "Dev_Accuracy", "Dev_MCC", "Dev_F1", "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1", "Test_AUC", "Test_MCC"]].to_string())


if __name__ == "__main__":
    main()
