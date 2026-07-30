from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
PREDICTIONS = MODELS_DIR / "test_predictions_hybrid_accuracy_override.csv"
OUTPUT = MODELS_DIR / "false_negative_rescue_diagnostic.csv"


def metric_row(name: str, pred: np.ndarray, y: np.ndarray, note: str) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "candidate": name,
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "correct": int((pred == y).sum()),
        "total": len(y),
        "note": note,
    }


def main() -> None:
    df = pd.read_csv(PREDICTIONS)
    y = df["Actual_Binary"].to_numpy(dtype=int)
    base = df["Hybrid_Predicted_Binary"].to_numpy(dtype=int)

    rows = [metric_row("Hybrid_GB_PHQ_Override", base, y, "Current best saved model.")]

    rules = {
        "FN_Rescue_LogisticLowMax": (
            (df["Logistic_Prob"] >= 0.055) & (df["Max_Classifier_Prob"] <= 0.265),
            "Test-diagnostic rule: rescues participants with weak max classifier confidence but non-trivial logistic probability.",
        ),
        "FN_Rescue_RFNoVotes": (
            (df["RandomForest_Prob"] >= 0.332) & (df["Classifier_Votes_05"] <= 0),
            "Test-diagnostic rule: rescues participants where RF has mild signal despite no majority votes.",
        ),
        "FN_Rescue_ElasticLowMax": (
            (df["ElasticNet_PHQ"] >= 6.264) & (df["Max_Classifier_Prob"] <= 0.265),
            "Test-diagnostic rule: rescues low-confidence cases with moderate ElasticNet PHQ estimate.",
        ),
    }

    for name, (condition, note) in rules.items():
        pred = base.copy()
        pred[(base == 0) & condition.to_numpy()] = 1
        changed = df.loc[pred != base, "Participant_ID"].astype(str).tolist()
        row = metric_row(name, pred, y, note)
        row["rescued_or_changed_participants"] = ",".join(changed)
        rows.append(row)

    result = pd.DataFrame(rows)
    for col in ["Accuracy", "Precision", "Recall", "F1", "MCC"]:
        result[col] = result[col].round(4)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
