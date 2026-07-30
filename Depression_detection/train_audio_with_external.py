from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent
LABELS_DIR = BASE_DIR / "labels"
DAIC_AUDIO_DIR = BASE_DIR / "preprocessed_audio"
EXTERNAL_DIR = BASE_DIR / "external_training_data"
EXTERNAL_AUDIO_DIR = EXTERNAL_DIR / "preprocessed_audio"
EXTERNAL_MANIFEST = EXTERNAL_DIR / "external_audio_manifest.csv"
MODELS_DIR = BASE_DIR / "models"
FEATURE_DIM = 153
RANDOM_SEED = 42


def load_daic_labels(csv_name: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    df = pd.read_csv(LABELS_DIR / csv_name)
    pid_col = next(c for c in df.columns if "participant_id" in c.lower())
    score_col = next(c for c in df.columns if "score" in c.lower())
    bin_col = next((c for c in df.columns if "binary" in c.lower()), None)
    ids = [str(int(pid)) for pid in df[pid_col]]
    scores = df[score_col].to_numpy(dtype=float)
    if bin_col:
        labels = df[bin_col].to_numpy(dtype=int)
    else:
        labels = (scores >= 10).astype(int)
    return ids, scores, labels


def load_vector(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    vector = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    fixed = np.zeros(FEATURE_DIM, dtype=np.float32)
    fixed[: min(FEATURE_DIM, vector.size)] = vector[:FEATURE_DIM]
    return np.nan_to_num(fixed)


def load_daic_split(csv_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    ids, scores, labels = load_daic_labels(csv_name)
    missing = 0
    rows = []
    for pid in ids:
        path = DAIC_AUDIO_DIR / f"{pid}_audio_features.npy"
        missing += int(not path.exists())
        rows.append(load_vector(path))
    return np.vstack(rows), scores, labels, missing


def load_external_audio() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    manifest = pd.read_csv(EXTERNAL_MANIFEST)
    rows = []
    scores = []
    labels = []
    for _, row in manifest.iterrows():
        participant_id = str(row["participant_id"])
        path = EXTERNAL_AUDIO_DIR / f"{participant_id}_audio_features.npy"
        rows.append(load_vector(path))
        scores.append(float(row["phq_score"]))
        labels.append(int(row["phq_binary"]))
    return np.vstack(rows), np.asarray(scores), np.asarray(labels)


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray, objective: str) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_key = (-1.0, -1.0, -1.0)
    best = {}
    for threshold in np.arange(0.05, 0.96, 0.01):
        pred = (probabilities >= threshold).astype(int)
        accuracy = accuracy_score(y_true, pred)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, pred)
        if objective == "accuracy":
            key = (accuracy, mcc, f1)
        elif objective == "f1":
            key = (f1, accuracy, mcc)
        else:
            key = (recall, f1, accuracy)
        if key > best_key:
            best_threshold = float(threshold)
            best_key = key
            best = {
                "Dev_Accuracy": accuracy,
                "Dev_Precision": precision,
                "Dev_Recall": recall,
                "Dev_F1": f1,
                "Dev_MCC": mcc,
            }
    return best_threshold, best


def evaluate(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
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


def main() -> None:
    x_daic_train, _, y_daic_train, missing_train = load_daic_split("train_split_Depression_AVEC.csv")
    x_dev, _, y_dev, missing_dev = load_daic_split("dev_split_Depression_AVEC.csv")
    x_test, _, y_test, missing_test = load_daic_split("full_test_split.csv")
    x_ext, _, y_ext = load_external_audio()

    train_variants = {
        "DAIC_only": (x_daic_train, y_daic_train),
        "DAIC_plus_external_audio": (np.vstack([x_daic_train, x_ext]), np.concatenate([y_daic_train, y_ext])),
    }
    scalers = {
        "standard": StandardScaler(),
        "robust": RobustScaler(),
    }
    model_factories = {
        "AudioExt_LogisticRegression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED),
        "AudioExt_RandomForest": lambda: RandomForestClassifier(n_estimators=700, class_weight="balanced_subsample", random_state=RANDOM_SEED),
        "AudioExt_ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=900, class_weight="balanced", random_state=RANDOM_SEED),
        "AudioExt_GradientBoosting": lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=0.06, max_depth=2, random_state=RANDOM_SEED),
        "AudioExt_SVC_RBF": lambda: SVC(C=1.2, gamma="scale", class_weight="balanced", probability=True, random_state=RANDOM_SEED),
    }

    rows = []
    for train_name, (x_train_raw, y_train) in train_variants.items():
        for scaler_name, scaler in scalers.items():
            x_train = scaler.fit_transform(x_train_raw)
            x_dev_s = scaler.transform(x_dev)
            x_test_s = scaler.transform(x_test)
            fitted_models = {}
            for model_name, factory in model_factories.items():
                model = factory()
                model.fit(x_train, y_train)
                fitted_models[model_name] = model
                dev_prob = model.predict_proba(x_dev_s)[:, 1]
                test_prob = model.predict_proba(x_test_s)[:, 1]
                for objective in ["accuracy", "f1", "recall"]:
                    threshold, dev = select_threshold(y_dev, dev_prob, objective)
                    row = {
                        "training_data": train_name,
                        "model": model_name,
                        "scaler": scaler_name,
                        "objective": objective,
                        "threshold": threshold,
                        **dev,
                        **evaluate(y_test, test_prob, threshold),
                        "daic_train_rows": len(x_daic_train),
                        "external_audio_rows": len(x_ext) if train_name != "DAIC_only" else 0,
                        "missing_daic_train_audio": missing_train,
                        "missing_daic_dev_audio": missing_dev,
                        "missing_daic_test_audio": missing_test,
                    }
                    rows.append(row)
                if train_name == "DAIC_plus_external_audio" and scaler_name == "robust":
                    joblib.dump(model, MODELS_DIR / f"{model_name}_external_audio_classifier.pkl")

            ensemble = VotingClassifier(
                estimators=[(name.replace("AudioExt_", "").lower(), factory()) for name, factory in model_factories.items()],
                voting="soft",
                weights=[1, 1, 1, 2, 2],
            )
            ensemble.fit(x_train, y_train)
            dev_prob = ensemble.predict_proba(x_dev_s)[:, 1]
            test_prob = ensemble.predict_proba(x_test_s)[:, 1]
            for objective in ["accuracy", "f1", "recall"]:
                threshold, dev = select_threshold(y_dev, dev_prob, objective)
                rows.append({
                    "training_data": train_name,
                    "model": "AudioExt_SoftVotingEnsemble",
                    "scaler": scaler_name,
                    "objective": objective,
                    "threshold": threshold,
                    **dev,
                    **evaluate(y_test, test_prob, threshold),
                    "daic_train_rows": len(x_daic_train),
                    "external_audio_rows": len(x_ext) if train_name != "DAIC_only" else 0,
                    "missing_daic_train_audio": missing_train,
                    "missing_daic_dev_audio": missing_dev,
                    "missing_daic_test_audio": missing_test,
                })
            if train_name == "DAIC_plus_external_audio" and scaler_name == "robust":
                joblib.dump(ensemble, MODELS_DIR / "AudioExt_SoftVotingEnsemble_external_audio_classifier.pkl")
                joblib.dump(scaler, MODELS_DIR / "AudioExt_RobustScaler_external_audio.pkl")

    result = pd.DataFrame(rows).sort_values(["Accuracy", "MCC", "F1"], ascending=False)
    output = MODELS_DIR / "audio_external_training_comparison.csv"
    result.to_csv(output, index=False)
    print(result.head(12).round(4).to_string(index=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
