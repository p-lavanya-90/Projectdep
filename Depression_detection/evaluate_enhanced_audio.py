"""Evaluate COVAREP/Formant-enhanced audio with fixed Logistic Regression."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, fbeta_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

SEED = 42
AUDIO_DIM, IMAGE_DIM, TEXT_DIM = 311, 160, 768
BASE = Path(__file__).resolve().parent
LABELS, MODELS = BASE / "labels", BASE / "models"
AUDIO, IMAGE, TEXT = BASE / "preprocessed_audio_enhanced", BASE / "preprocessed_images", BASE / "preprocessed_text"


def load_vector(folder, participant_id, suffix, dimension):
    path = folder / f"{participant_id}_{suffix}.npy"
    if not path.exists():
        return np.zeros(dimension, dtype=np.float32)
    vector = np.load(path).astype(np.float32).flatten()
    if vector.size != dimension:
        raise ValueError(f"{path.name}: expected {dimension}, got {vector.size}")
    return vector


def load_split(filename):
    frame = pd.read_csv(LABELS / filename)
    frame.columns = [column.lower() for column in frame.columns]
    pid_col = next(column for column in frame.columns if "participant_id" in column)
    score_col = next(column for column in frame.columns if "score" in column)
    binary_col = next((column for column in frame.columns if "binary" in column), None)
    rows, labels = [], []
    for _, row in frame.iterrows():
        participant_id = str(int(row[pid_col]))
        rows.append(np.concatenate([
            load_vector(AUDIO, participant_id, "audio_features", AUDIO_DIM),
            load_vector(IMAGE, participant_id, "image_features", IMAGE_DIM),
            load_vector(TEXT, participant_id, "text_features", TEXT_DIM),
        ]))
        labels.append(int(row[binary_col]) if binary_col and not pd.isna(row[binary_col]) else int(row[score_col] >= 10))
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=int)


def select_threshold(labels, probabilities):
    best_threshold, best_f2 = 0.50, -1.0
    for threshold in np.arange(0.05, 0.96, 0.01):
        score = fbeta_score(labels, probabilities >= threshold, beta=2.0, zero_division=0)
        if score > best_f2:
            best_threshold, best_f2 = float(threshold), float(score)
    return best_threshold, best_f2


def evaluate(name, columns, x_train, x_dev, x_test, y_train, y_dev, y_test):
    scaler = StandardScaler()
    train = scaler.fit_transform(x_train[:, columns])
    dev = scaler.transform(x_dev[:, columns])
    test = scaler.transform(x_test[:, columns])
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    model.fit(train, y_train)
    threshold, dev_f2 = select_threshold(y_dev, model.predict_proba(dev)[:, 1])
    probabilities = model.predict_proba(test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_test, predictions)
    return {
        "experiment": name, "feature_dimension": train.shape[1],
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "Accuracy": accuracy_score(y_test, predictions), "AUC": roc_auc_score(y_test, probabilities),
        "MCC": matthews_corrcoef(y_test, predictions), "Threshold": threshold, "Dev_F2": dev_f2,
        "TN": cm[0, 0], "FP": cm[0, 1], "FN": cm[1, 0], "TP": cm[1, 1],
    }


def main():
    x_train, y_train = load_split("train_split_Depression_AVEC.csv")
    x_dev, y_dev = load_split("dev_split_Depression_AVEC.csv")
    x_test, y_test = load_split("full_test_split.csv")
    experiments = {
        "Enhanced audio only": slice(0, AUDIO_DIM),
        "Text + enhanced audio": np.r_[0:AUDIO_DIM, AUDIO_DIM + IMAGE_DIM:AUDIO_DIM + IMAGE_DIM + TEXT_DIM],
        "Text + enhanced audio + visual CLNF": slice(0, AUDIO_DIM + IMAGE_DIM + TEXT_DIM),
    }
    results = [evaluate(name, columns, x_train, x_dev, x_test, y_train, y_dev, y_test) for name, columns in experiments.items()]
    output = MODELS / "enhanced_audio_comparison.csv"
    pd.DataFrame(results).to_csv(output, index=False)
    print(pd.DataFrame(results).round(4).to_string(index=False))
    print(f"Saved comparison: {output}")


if __name__ == "__main__":
    main()
