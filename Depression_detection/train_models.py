import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    fbeta_score, roc_auc_score, matthews_corrcoef, confusion_matrix,
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Select the depression decision threshold on the development split.  F2 gives
# recall twice the importance of precision, which reduces costly false negatives.
THRESHOLD_BETA = 2.0

# Paths
# Resolve paths from this script so the pipeline works on any machine.
base_dir = Path(__file__).resolve().parent
labels_dir = base_dir / "labels"
preprocessed_dirs = [
    base_dir / "preprocessed_audio",
    base_dir / "preprocessed_images",
    base_dir / "preprocessed_text"
]
models_dir = base_dir / "models"
models_dir.mkdir(exist_ok=True)

# MLP Regressor (PyTorch)
class MLPRegressorModel(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegressorModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def load_features(p_id, folder, prefix):
    f_path = folder / f"{p_id}_{prefix}.npy"
    if f_path.exists():
        return np.load(f_path)
    # Default dimensions
    dims = {"audio": 153, "image": 160, "text": 768}
    return np.zeros(dims.get(prefix.split("_")[0], 0))

def prepare_dataset(csv_file, preprocessed_dirs):
    if not csv_file.exists():
        print(f"Warning: CSV file {csv_file.name} not found.")
        return None, None, None
        
    df = pd.read_csv(csv_file)
    df.columns = [c.lower() for c in df.columns]
    
    X, y_score, y_binary = [], [], []
    audio_dir, image_dir, text_dir = preprocessed_dirs
    
    # Identify standard column names
    pid_col = next((c for c in df.columns if 'participant_id' in c), None)
    score_col = next((c for c in df.columns if 'score' in c), None)
    bin_col = next((c for c in df.columns if 'binary' in c), None)

    if not pid_col or not score_col:
        print(f"Error: Missing required columns in {csv_file.name}")
        return None, None, None

    for _, row in df.iterrows():
        p_id = str(int(row[pid_col]))
        
        # Load preprocessed features
        a_feat = load_features(p_id, audio_dir, "audio_features")
        i_feat = load_features(p_id, image_dir, "image_features")
        t_feat = load_features(p_id, text_dir, "text_features")
        
        # Fusion: Concatenation
        combined = np.hstack([a_feat.flatten(), i_feat.flatten(), t_feat.flatten()])
        X.append(combined)
        
        # Label: PHQ-8 Score
        score = row[score_col]
        y_score.append(score)
        
        # Label: Secondary Classification (Score >= 10)
        # Note: If bin_col exists, use it, otherwise use >= 10 logic.
        if bin_col and not pd.isna(row[bin_col]):
            y_binary.append(row[bin_col])
        else:
            y_binary.append(1 if score >= 10 else 0)
            
    return np.array(X), np.array(y_score), np.array(y_binary)


def select_depression_threshold(y_true, probabilities, beta=THRESHOLD_BETA):
    """Choose a threshold using only development data, never the test split."""
    best_threshold, best_score = 0.50, -1.0
    for threshold in np.arange(0.05, 0.96, 0.01):
        predictions = (probabilities >= threshold).astype(int)
        score = fbeta_score(y_true, predictions, beta=beta, zero_division=0)
        # Prefer a lower threshold when scores tie: it favours depression recall.
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score


def run_ablation_study(X_train, X_dev, X_test, y_train, y_dev, y_test):
    """Measure each modality's contribution with one fixed classifier.

    Every experiment uses the same class-weighted Logistic Regression and the
    same development-set threshold-selection procedure. This makes changes in
    score attributable to the input modalities rather than to model choice.
    """
    modality_columns = {
        "Text only": slice(153 + 160, 153 + 160 + 768),
        "Audio only": slice(0, 153),
        "Visual CLNF only": slice(153, 153 + 160),
        "Text + Audio": np.r_[0:153, 153 + 160:153 + 160 + 768],
        "Text + Audio + Visual CLNF": slice(0, 153 + 160 + 768),
    }

    results = []
    for experiment, columns in modality_columns.items():
        x_train = X_train[:, columns]
        x_dev = X_dev[:, columns]
        x_test = X_test[:, columns]

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_dev = scaler.transform(x_dev)
        x_test = scaler.transform(x_test)

        model = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
        )
        model.fit(x_train, y_train)
        dev_probabilities = model.predict_proba(x_dev)[:, 1]
        threshold, dev_f2 = select_depression_threshold(y_dev, dev_probabilities)
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        cm = confusion_matrix(y_test, predictions)

        row = {
            "experiment": experiment,
            "feature_dimension": x_train.shape[1],
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1": f1_score(y_test, predictions, zero_division=0),
            "Accuracy": accuracy_score(y_test, predictions),
            "AUC": roc_auc_score(y_test, probabilities),
            "MCC": matthews_corrcoef(y_test, predictions),
            "Threshold": threshold,
            "Dev_F2": dev_f2,
            "TN": cm[0, 0], "FP": cm[0, 1],
            "FN": cm[1, 0], "TP": cm[1, 1],
        }
        results.append(row)
        print(
            f"{experiment:28} | F1: {row['F1']:.3f} | Recall: {row['Recall']:.3f} | "
            f"AUC: {row['AUC']:.3f} | MCC: {row['MCC']:.3f}"
        )

    output = models_dir / "ablation_comparison.csv"
    pd.DataFrame(results).to_csv(output, index=False)
    print(f"Ablation comparison saved to: {output}")

def train_mlp_reg(X_train, y_train, X_val, y_val, input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRegressorModel(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    
    best_loss = float('inf')
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_t)
        loss = criterion(outputs, y_t)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_v)
                val_loss = criterion(val_outputs, y_v).item()
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), models_dir / "MLP_regressor.pth")
    return model

def run_evaluation():
    print("=== Multimodal Pipeline Training ===")
    X_train, y_train_scr, y_train_bin = prepare_dataset(labels_dir / "train_split_Depression_AVEC.csv", preprocessed_dirs)
    X_dev, y_dev_scr, y_dev_bin = prepare_dataset(labels_dir / "dev_split_Depression_AVEC.csv", preprocessed_dirs)
    X_test, y_test_scr, y_test_bin = prepare_dataset(labels_dir / "full_test_split.csv", preprocessed_dirs)

    if X_train is None or X_dev is None or X_test is None:
        print("Missing dataset files. Ensure filenames match 'labels/' content.")
        return
        
    print(f"Records: Train={len(X_train)}, Dev={len(X_dev)}, Test={len(X_test)}")
    
    # 2. Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_dev_s   = scaler.transform(X_dev)
    X_test_s  = scaler.transform(X_test)
    joblib.dump(scaler, models_dir / "feature_scaler.pkl")

    # 3. Primary Task: REGRESSION
    print("\n--- Training Regression Models ---")
    reg_models = {
        "LinearRegression": LinearRegression(),
        "ElasticNet": ElasticNet(random_state=RANDOM_SEED),
        "RandomForestReg": RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED),
        "XGBoostReg": XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=RANDOM_SEED),
        "LightGBMReg": LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=RANDOM_SEED, verbose=-1)
    }
    
    reg_results = []
    for name, model in reg_models.items():
        model.fit(X_train_s, y_train_scr)
        preds = model.predict(X_test_s)
        mae = mean_absolute_error(y_test_scr, preds)
        rmse = np.sqrt(mean_squared_error(y_test_scr, preds))
        r2 = r2_score(y_test_scr, preds)
        reg_results.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        joblib.dump(model, models_dir / f"{name}_regressor.pkl")
        print(f"{name:16} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")

    # MLP Deep Learning Regression
    mlp_model = train_mlp_reg(X_train_s, y_train_scr, X_dev_s, y_dev_scr, X_train.shape[1])
    mlp_model.eval()
    with torch.no_grad():
        mlp_preds = mlp_model(torch.tensor(X_test_s, dtype=torch.float32)).cpu().numpy().flatten()
    mae_mlp = mean_absolute_error(y_test_scr, mlp_preds)
    rmse_mlp = np.sqrt(mean_squared_error(y_test_scr, mlp_preds))
    r2_mlp = r2_score(y_test_scr, mlp_preds)
    reg_results.append({"model": "DeepMLP", "MAE": mae_mlp, "RMSE": rmse_mlp, "R2": r2_mlp})
    print(f"DeepMLP          | MAE: {mae_mlp:.3f} | RMSE: {rmse_mlp:.3f} | R2: {r2_mlp:.3f}")
    
    # Save regression comparison
    pd.DataFrame(reg_results).to_csv(models_dir / "regression_comparison.csv", index=False)

    # 4. Secondary Task: CLASSIFICATION
    print("\n--- Training Classification Models ---")
    negative_count = int(np.sum(y_train_bin == 0))
    positive_count = int(np.sum(y_train_bin == 1))
    if positive_count == 0:
        raise ValueError("The training split contains no depressed samples.")
    scale_pos_weight = negative_count / positive_count
    print(
        f"Class balance: non-depressed={negative_count}, depressed={positive_count}; "
        f"XGBoost scale_pos_weight={scale_pos_weight:.3f}"
    )

    clf_models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "RandomForestClf": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", random_state=RANDOM_SEED
        ),
        "XGBoostClf": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=RANDOM_SEED
        ),
    }
    
    clf_results = []
    for name, model in clf_models.items():
        model.fit(X_train_s, y_train_bin)
        dev_probs = model.predict_proba(X_dev_s)[:, 1]
        threshold, dev_f2 = select_depression_threshold(y_dev_bin, dev_probs)
        probs = model.predict_proba(X_test_s)[:, 1]
        preds = (probs >= threshold).astype(int)

        precision = precision_score(y_test_bin, preds, zero_division=0)
        recall = recall_score(y_test_bin, preds, zero_division=0)
        f1 = f1_score(y_test_bin, preds, zero_division=0)
        acc = accuracy_score(y_test_bin, preds)
        auc = roc_auc_score(y_test_bin, probs)
        mcc = matthews_corrcoef(y_test_bin, preds)
        cm = confusion_matrix(y_test_bin, preds)

        clf_results.append({
            "model": name,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Accuracy": acc,
            "AUC": auc,
            "MCC": mcc,
            "Threshold": threshold,
            "Dev_F2": dev_f2,
            "TN": cm[0, 0],
            "FP": cm[0, 1],
            "FN": cm[1, 0],
            "TP": cm[1, 1],
        })
        joblib.dump(model, models_dir / f"{name}_classifier.pkl")
        print(
            f"{name:16} | Precision: {precision:.3f} | Recall: {recall:.3f} | "
            f"F1: {f1:.3f} | Acc: {acc:.3f} | AUC: {auc:.3f} | MCC: {mcc:.3f} | "
            f"Threshold: {threshold:.2f}"
        )

    # Save the complete held-out test-set classification comparison.
    pd.DataFrame(clf_results).to_csv(models_dir / "classification_comparison.csv", index=False)

    print("\n--- Modality Ablation Study (fixed Logistic Regression) ---")
    run_ablation_study(
        X_train, X_dev, X_test,
        y_train_bin, y_dev_bin, y_test_bin,
    )
    
    # Save processed datasets for analysis
    final_dataset_dir = base_dir / "final_dataset"
    final_dataset_dir.mkdir(exist_ok=True)
    np.save(final_dataset_dir / "X_train.npy", X_train_s)
    np.save(final_dataset_dir / "X_dev.npy", X_dev_s)
    np.save(final_dataset_dir / "X_test.npy", X_test_s)
    np.save(final_dataset_dir / "y_train_score.npy", y_train_scr)
    np.save(final_dataset_dir / "y_dev_score.npy", y_dev_scr)
    np.save(final_dataset_dir / "y_test_score.npy", y_test_scr)
    np.save(final_dataset_dir / "y_train_binary.npy", y_train_bin)
    np.save(final_dataset_dir / "y_dev_binary.npy", y_dev_bin)
    np.save(final_dataset_dir / "y_test_binary.npy", y_test_bin)

    print("\nPipeline execution complete. Models and comparisons saved in 'models/'.")

if __name__ == "__main__":
    run_evaluation()
