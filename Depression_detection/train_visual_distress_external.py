from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent
EXTERNAL_DIR = BASE_DIR / "external_training_data"
IMAGE_ROOT = EXTERNAL_DIR / "DepVidMood" / "Depression Data" / "data"
MODELS_DIR = BASE_DIR / "models"
FEATURE_DIM = 160
RANDOM_SEED = 42

DISTRESS_EMOTIONS = {"Angry", "Disgust", "Fear", "Sad"}
NON_DISTRESS_EMOTIONS = {"Happy", "Neutral", "Surprize"}


def image_features(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize((48, 48))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    hist, _ = np.histogram(arr, bins=64, range=(0.0, 1.0), density=True)
    row_mean = arr.mean(axis=1)
    col_mean = arr.mean(axis=0)
    row_std = arr.std(axis=1)
    col_std = arr.std(axis=0)
    stat = ImageStat.Stat(image)
    global_stats = np.asarray(
        [
            arr.mean(),
            arr.std(),
            arr.min(),
            arr.max(),
            np.median(arr),
            np.percentile(arr, 25),
            np.percentile(arr, 75),
            stat.rms[0] / 255.0,
        ],
        dtype=np.float32,
    )
    vector = np.concatenate([hist, row_mean, col_mean, row_std, col_std, global_stats]).astype(np.float32)
    fixed = np.zeros(FEATURE_DIM, dtype=np.float32)
    fixed[: min(FEATURE_DIM, vector.size)] = vector[:FEATURE_DIM]
    if not np.isfinite(fixed).all():
        raise ValueError(f"Non-finite image features: {path}")
    return fixed


def label_for_emotion(emotion: str) -> int:
    if emotion in DISTRESS_EMOTIONS:
        return 1
    if emotion in NON_DISTRESS_EMOTIONS:
        return 0
    raise ValueError(f"Unknown emotion folder: {emotion}")


def load_split(split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows = []
    labels = []
    metadata = []
    split_dir = IMAGE_ROOT / split
    for emotion_dir in sorted(split_dir.iterdir()):
        if not emotion_dir.is_dir():
            continue
        label = label_for_emotion(emotion_dir.name)
        for path in sorted(emotion_dir.glob("*.png")):
            rows.append(image_features(path))
            labels.append(label)
            metadata.append(
                {
                    "split": split,
                    "emotion": emotion_dir.name,
                    "distress_label": label,
                    "path": path,
                }
            )
    return np.vstack(rows), np.asarray(labels), pd.DataFrame(metadata)


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
                "Val_Accuracy": accuracy,
                "Val_Precision": precision,
                "Val_Recall": recall,
                "Val_F1": f1,
                "Val_MCC": mcc,
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
    x_train, y_train, train_meta = load_split("train")
    x_val, y_val, val_meta = load_split("val")
    x_test, y_test, test_meta = load_split("test")
    pd.concat([train_meta, val_meta, test_meta], ignore_index=True).to_csv(
        EXTERNAL_DIR / "depvidmood_image_manifest.csv", index=False
    )

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)

    model_factories = {
        "DepVidMood_LogisticRegression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED),
        "DepVidMood_RandomForest": lambda: RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=RANDOM_SEED),
        "DepVidMood_ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=700, class_weight="balanced", random_state=RANDOM_SEED),
        "DepVidMood_GradientBoosting": lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=0.06, max_depth=2, random_state=RANDOM_SEED),
        "DepVidMood_SVC_RBF": lambda: SVC(C=1.2, gamma="scale", class_weight="balanced", probability=True, random_state=RANDOM_SEED),
    }

    rows = []
    for name, factory in model_factories.items():
        model = factory()
        model.fit(x_train_s, y_train)
        val_prob = model.predict_proba(x_val_s)[:, 1]
        test_prob = model.predict_proba(x_test_s)[:, 1]
        for objective in ["accuracy", "f1", "recall"]:
            threshold, val = select_threshold(y_val, val_prob, objective)
            rows.append({
                "model": name,
                "objective": objective,
                "threshold": threshold,
                **val,
                **evaluate(y_test, test_prob, threshold),
                "train_rows": len(y_train),
                "val_rows": len(y_val),
                "test_rows": len(y_test),
            })
        joblib.dump(model, MODELS_DIR / f"{name}_visual_distress_classifier.pkl")

    ensemble = VotingClassifier(
        estimators=[(name.replace("DepVidMood_", "").lower(), factory()) for name, factory in model_factories.items()],
        voting="soft",
        weights=[1, 1, 1, 2, 2],
    )
    ensemble.fit(x_train_s, y_train)
    val_prob = ensemble.predict_proba(x_val_s)[:, 1]
    test_prob = ensemble.predict_proba(x_test_s)[:, 1]
    for objective in ["accuracy", "f1", "recall"]:
        threshold, val = select_threshold(y_val, val_prob, objective)
        rows.append({
            "model": "DepVidMood_SoftVotingEnsemble",
            "objective": objective,
            "threshold": threshold,
            **val,
            **evaluate(y_test, test_prob, threshold),
            "train_rows": len(y_train),
            "val_rows": len(y_val),
            "test_rows": len(y_test),
        })
    joblib.dump(ensemble, MODELS_DIR / "DepVidMood_SoftVotingEnsemble_visual_distress_classifier.pkl")
    joblib.dump(scaler, MODELS_DIR / "DepVidMood_VisualDistressScaler.pkl")

    result = pd.DataFrame(rows).sort_values(["Accuracy", "MCC", "F1"], ascending=False)
    output = MODELS_DIR / "depvidmood_visual_distress_comparison.csv"
    result.to_csv(output, index=False)
    print(result.head(12).round(4).to_string(index=False))
    print(f"Wrote {output}")
    print(f"Wrote {EXTERNAL_DIR / 'depvidmood_image_manifest.csv'}")


if __name__ == "__main__":
    main()
