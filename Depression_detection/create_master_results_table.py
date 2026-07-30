from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_CSV = MODELS_DIR / "master_project_results_table.csv"
OUTPUT_MD = MODELS_DIR / "master_project_results_table.md"

METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUC", "MCC"]
BASE_VALUES = {
    "Accuracy": 0.85,
    "Precision": 0.73,
    "Recall": 0.85,
    "F1": 0.79,
    "AUC": 0.73,
    "MCC": 0.68,
}


def metric_status(row: pd.Series) -> str:
    if row["Category"] == "Base Paper":
        return "Reference baseline"
    if str(row["Category"]).startswith("Auxiliary Visual"):
        return "Auxiliary dataset, not DAIC-comparable"
    beaten = []
    close = []
    for metric in METRICS:
        value = row.get(metric)
        if pd.isna(value):
            continue
        if float(value) > BASE_VALUES[metric]:
            beaten.append(metric)
        elif abs(float(value) - BASE_VALUES[metric]) <= 0.01:
            close.append(metric)
    parts = []
    if beaten:
        parts.append("Beats: " + ", ".join(beaten))
    if close:
        parts.append("Close: " + ", ".join(close))
    return "; ".join(parts) if parts else "Below base paper"


def pick_best(path: Path, sort_cols: list[str], category: str, purpose: str, name_col: str = "model") -> dict[str, object]:
    df = pd.read_csv(path)
    row = df.sort_values(sort_cols, ascending=False).iloc[0]
    return {
        "Category": category,
        "Model/System": row[name_col],
        "Purpose": purpose,
        "Accuracy": row["Accuracy"],
        "Precision": row["Precision"],
        "Recall": row["Recall"],
        "F1": row["F1"],
        "AUC": row["AUC"],
        "MCC": row["MCC"],
        "Threshold": row.get("Threshold", ""),
        "Notes": "",
    }


def pick_external_audio() -> dict[str, object] | None:
    path = MODELS_DIR / "audio_external_training_comparison.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    external = df.loc[df["training_data"] == "DAIC_plus_external_audio"].copy()
    if external.empty:
        return None

    row = external.sort_values(["Accuracy", "MCC", "F1"], ascending=False).iloc[0]
    return {
        "Category": "External Audio",
        "Model/System": row["model"],
        "Purpose": "Best DAIC + external audio branch",
        "Accuracy": row["Accuracy"],
        "Precision": row["Precision"],
        "Recall": row["Recall"],
        "F1": row["F1"],
        "AUC": row["AUC"],
        "MCC": row["MCC"],
        "Threshold": row.get("threshold", ""),
        "Notes": "External audio improved audio-only accuracy from 0.7021 to 0.7234 on the DAIC test split.",
    }


def pick_visual_cnn() -> dict[str, object] | None:
    path = MODELS_DIR / "depvidmood_cnn_visual_distress_comparison.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    test_rows = df.loc[df["split"] == "test"]
    row = (test_rows if not test_rows.empty else df).iloc[0]
    return {
        "Category": "Auxiliary Visual CNN",
        "Model/System": "DepVidMood_CNN_visual_distress",
        "Purpose": "Raw-image expression distress fallback",
        "Accuracy": row["Accuracy"],
        "Precision": row["Precision"],
        "Recall": row["Recall"],
        "F1": row["F1"],
        "AUC": row["AUC"],
        "MCC": row["MCC"],
        "Threshold": "",
        "Notes": "DepVidMood test split; supports visual website fallback, not a DAIC PHQ-8 depression metric.",
    }


def main() -> None:
    rows: list[dict[str, object]] = []

    rows.append({
        "Category": "Base Paper",
        "Model/System": "Base Paper",
        "Purpose": "Reported baseline",
        **BASE_VALUES,
        "Threshold": "",
        "Notes": "Reference values from base paper",
    })

    final = pd.read_csv(MODELS_DIR / "final_base_paper_comparison.csv")
    for name in [
        "Hybrid_GB_PHQ_Override",
        "RecallOptimized_Logistic_RF_Override",
        "F1Optimized_Logistic_AND_GB",
        "GradientBoostingClf_Accuracy",
    ]:
        row = final.loc[final["Model/System"] == name].iloc[0]
        rows.append({
            "Category": "Multimodal",
            "Model/System": row["Model/System"],
            "Purpose": row["Purpose"],
            "Accuracy": row["Accuracy"],
            "Precision": row["Precision"],
            "Recall": row["Recall"],
            "F1": row["F1"],
            "AUC": row["AUC"],
            "MCC": row["MCC"],
            "Threshold": "",
            "Notes": row.get("Status_vs_Base", ""),
        })

    ensemble_summary = pd.read_csv(MODELS_DIR / "accuracy_ensemble_search_summary.csv")
    dev_ensemble = ensemble_summary.loc[ensemble_summary["selection"] == "best_by_dev"].iloc[0]
    rows.append({
        "Category": "Multimodal Ensemble",
        "Model/System": dev_ensemble["candidate"],
        "Purpose": "Best dev-selected balanced ensemble",
        "Accuracy": dev_ensemble["Test_Accuracy"],
        "Precision": dev_ensemble["Test_Precision"],
        "Recall": dev_ensemble["Test_Recall"],
        "F1": dev_ensemble["Test_F1"],
        "AUC": dev_ensemble["Test_AUC"],
        "MCC": dev_ensemble["Test_MCC"],
        "Threshold": dev_ensemble["threshold"],
        "Notes": dev_ensemble["weights"],
    })

    rows.append(pick_best(
        MODELS_DIR / "text_only_benchmark.csv",
        ["Accuracy", "MCC", "F1"],
        "Text Only",
        "Best text-only accuracy from current embeddings",
    ))
    rows.append(pick_best(
        MODELS_DIR / "audio_feature_benchmark.csv",
        ["Accuracy", "MCC", "F1"],
        "Audio Only",
        "Best audio-only accuracy from current 153-dim audio features",
    ))
    external_audio = pick_external_audio()
    if external_audio:
        rows.append(external_audio)
    rows.append(pick_best(
        MODELS_DIR / "visual_feature_benchmark.csv",
        ["Accuracy", "MCC", "F1"],
        "Visual Only",
        "Best visual-only accuracy from CLNF/OpenFace features",
    ))
    visual_cnn = pick_visual_cnn()
    if visual_cnn:
        rows.append(visual_cnn)
    rows.append(pick_best(
        BASE_DIR / "improved_dataset" / "improved_accuracy_model_comparison.csv",
        ["Accuracy", "MCC", "F1"],
        "Improved Dataset",
        "Best clean dataset-improvement run",
    ))

    result = pd.DataFrame(rows)
    for metric in METRICS:
        result[metric] = pd.to_numeric(result[metric], errors="coerce").round(4)
    result["Status_vs_Base"] = result.apply(metric_status, axis=1)
    result = result[
        [
            "Category", "Model/System", "Purpose",
            "Accuracy", "Precision", "Recall", "F1", "AUC", "MCC",
            "Threshold", "Status_vs_Base", "Notes",
        ]
    ]
    result.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_MD.write_text(to_markdown(result) + "\n")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_MD}")
    print(result.to_string(index=False))


def to_markdown(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for header in headers:
            value = row[header]
            if pd.isna(value):
                value = ""
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
