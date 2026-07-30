from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent
LABELS_DIR = BASE_DIR / "labels"
AUDIO_DIR = BASE_DIR / "preprocessed_audio"
MODELS_DIR = BASE_DIR / "models"
FEATURE_DIM = 153
RANDOM_SEED = 42


def labels(csv_name: str) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(LABELS_DIR / csv_name)
    pid_col = next(c for c in df.columns if "participant_id" in c.lower())
    score_col = next(c for c in df.columns if "score" in c.lower())
    bin_col = next((c for c in df.columns if "binary" in c.lower()), None)
    ids = [str(int(pid)) for pid in df[pid_col]]
    if bin_col:
        binary = df[bin_col].to_numpy(dtype=int)
    else:
        binary = (df[score_col].to_numpy(dtype=float) >= 10).astype(int)
    return ids, binary


def load_audio(ids: list[str]) -> tuple[np.ndarray, int]:
    rows = []
    missing = 0
    for pid in ids:
        path = AUDIO_DIR / f"{pid}_audio_features.npy"
        if path.exists():
            vector = np.asarray(np.load(path), dtype=float).reshape(-1)
            fixed = np.zeros(FEATURE_DIM, dtype=float)
            fixed[: min(FEATURE_DIM, vector.size)] = vector[:FEATURE_DIM]
            rows.append(fixed)
        else:
            rows.append(np.zeros(FEATURE_DIM, dtype=float))
            missing += 1
    return np.vstack(rows), missing


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray, objective: str) -> tuple[float, tuple[float, float, float]]:
    best = (0.5, -1.0, -1.0, -1.0)
    for threshold in np.arange(0.05, 0.96, 0.01):
        pred = (probabilities >= threshold).astype(int)
        accuracy = accuracy_score(y_true, pred)
        mcc = matthews_corrcoef(y_true, pred)
        f1 = f1_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        if objective == "accuracy":
            key = (accuracy, mcc, f1)
        elif objective == "f1":
            key = (f1, accuracy, mcc)
        else:
            key = (recall, f1, accuracy)
        if key > best[1:]:
            best = (float(threshold), *key)
    return best[0], best[1:]


def main() -> None:
    train_ids, y_train = labels("train_split_Depression_AVEC.csv")
    dev_ids, y_dev = labels("dev_split_Depression_AVEC.csv")
    test_ids, y_test = labels("full_test_split.csv")
    x_train_raw, missing_train = load_audio(train_ids)
    x_dev_raw, missing_dev = load_audio(dev_ids)
    x_test_raw, missing_test = load_audio(test_ids)

    model_factories = {
        "Audio_LogisticRegression": lambda: LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_SEED),
        "Audio_RandomForest": lambda: RandomForestClassifier(n_estimators=700, class_weight="balanced_subsample", random_state=RANDOM_SEED),
        "Audio_ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=900, class_weight="balanced", random_state=RANDOM_SEED),
        "Audio_GradientBoosting": lambda: GradientBoostingClassifier(n_estimators=80, learning_rate=0.07, max_depth=2, random_state=RANDOM_SEED),
        "Audio_SVC_RBF": lambda: SVC(C=1.2, gamma="scale", class_weight="balanced", probability=True, random_state=RANDOM_SEED),
    }
    scalers = {
        "standard": StandardScaler(),
        "robust": RobustScaler(),
    }

    rows = []
    fitted_for_ensemble = []
    for scaler_name, scaler in scalers.items():
        x_train = scaler.fit_transform(x_train_raw)
        x_dev = scaler.transform(x_dev_raw)
        x_test = scaler.transform(x_test_raw)
        for model_name, factory in model_factories.items():
            model = factory()
            model.fit(x_train, y_train)
            for objective in ["accuracy", "f1", "recall"]:
                threshold, dev_key = select_threshold(y_dev, model.predict_proba(x_dev)[:, 1], objective)
                prob = model.predict_proba(x_test)[:, 1]
                pred = (prob >= threshold).astype(int)
                row = {
                    "model": model_name,
                    "scaler": scaler_name,
                    "objective": objective,
                    "Accuracy": accuracy_score(y_test, pred),
                    "Precision": precision_score(y_test, pred, zero_division=0),
                    "Recall": recall_score(y_test, pred, zero_division=0),
                    "F1": f1_score(y_test, pred, zero_division=0),
                    "AUC": roc_auc_score(y_test, prob),
                    "MCC": matthews_corrcoef(y_test, pred),
                    "Threshold": threshold,
                    "Dev_Key_1": dev_key[0],
                    "Dev_Key_2": dev_key[1],
                    "Dev_Key_3": dev_key[2],
                    "Missing_Train": missing_train,
                    "Missing_Dev": missing_dev,
                    "Missing_Test": missing_test,
                }
                rows.append(row)
            if scaler_name == "robust":
                fitted_for_ensemble.append((model_name, model, scaler))
                joblib.dump(model, MODELS_DIR / f"{model_name}_{scaler_name}_classifier.pkl")

    robust_scaler = RobustScaler()
    x_train = robust_scaler.fit_transform(x_train_raw)
    x_dev = robust_scaler.transform(x_dev_raw)
    x_test = robust_scaler.transform(x_test_raw)
    ensemble = VotingClassifier(
        estimators=[(name.replace("Audio_", "").lower(), factory()) for name, factory in model_factories.items()],
        voting="soft",
        weights=[1, 1, 1, 2, 2],
    )
    ensemble.fit(x_train, y_train)
    for objective in ["accuracy", "f1", "recall"]:
        threshold, dev_key = select_threshold(y_dev, ensemble.predict_proba(x_dev)[:, 1], objective)
        prob = ensemble.predict_proba(x_test)[:, 1]
        pred = (prob >= threshold).astype(int)
        rows.append({
            "model": "Audio_SoftVotingEnsemble",
            "scaler": "robust",
            "objective": objective,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "AUC": roc_auc_score(y_test, prob),
            "MCC": matthews_corrcoef(y_test, pred),
            "Threshold": threshold,
            "Dev_Key_1": dev_key[0],
            "Dev_Key_2": dev_key[1],
            "Dev_Key_3": dev_key[2],
            "Missing_Train": missing_train,
            "Missing_Dev": missing_dev,
            "Missing_Test": missing_test,
        })
    joblib.dump(ensemble, MODELS_DIR / "Audio_SoftVotingEnsemble_classifier.pkl")
    joblib.dump(robust_scaler, MODELS_DIR / "Audio_RobustScaler.pkl")

    result = pd.DataFrame(rows).sort_values(["Accuracy", "MCC", "F1"], ascending=False)
    output = MODELS_DIR / "audio_feature_benchmark.csv"
    result.to_csv(output, index=False)
    print(result.head(10).round(4).to_string(index=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
