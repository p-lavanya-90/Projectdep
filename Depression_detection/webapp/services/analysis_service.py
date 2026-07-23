import os, warnings
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from webapp.config import (X_TRAIN_BERT, X_DEV_BERT, Y_TRAIN_BIN, Y_DEV_BIN,
                            Y_TRAIN_SCR, Y_DEV_SCR, AUDIO_DIM, IMAGE_DIM, TEXT_DIM,
                            REG_COMP_CSV, CLF_COMP_CSV)

warnings.filterwarnings("ignore")

_cache = {}

def _load_data():
    if "X_train" in _cache:
        return _cache
    
    try:
        X_tr = np.load(X_TRAIN_BERT)
        X_dv = np.load(X_DEV_BERT)
        y_tr_bin = np.load(Y_TRAIN_BIN)
        y_dv_bin = np.load(Y_DEV_BIN)
        y_tr_scr = np.load(Y_TRAIN_SCR)
        y_dv_scr = np.load(Y_DEV_SCR)
        
        X_all   = np.vstack([X_tr, X_dv])
        y_all_b = np.concatenate([y_tr_bin, y_dv_bin])
        y_all_s = np.concatenate([y_tr_scr, y_dv_scr])

        _cache.update(dict(
            X_train=X_tr, X_dev=X_dv,
            y_train_bin=y_tr_bin, y_dev_bin=y_dv_bin,
            y_train_scr=y_tr_scr, y_dev_scr=y_dv_scr,
            X_all=X_all, y_all_bin=y_all_b, y_all_scr=y_all_s
        ))
        return _cache
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return None

def get_eda_stats() -> dict:
    d = _load_data()
    if not d: return {"error": "Dataset not ready."}
    
    y_tr, y_dv = d["y_train_bin"], d["y_dev_bin"]
    X_all, y_all = d["X_all"], d["y_all_bin"]

    # Slice modalities
    a_tr = d["X_train"][:, :AUDIO_DIM]
    i_tr = d["X_train"][:, AUDIO_DIM:AUDIO_DIM+IMAGE_DIM]
    t_tr = d["X_train"][:, AUDIO_DIM+IMAGE_DIM:]

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    pca = PCA(n_components=2, random_state=RANDOM_SEED if 'RANDOM_SEED' in globals() else 42)
    Xs  = StandardScaler().fit_transform(X_all[:, :min(X_all.shape[1], 100)]) # PCA on first 100 features
    pca_coords = pca.fit_transform(Xs)

    dep_idx = np.where(y_all == 1)[0].tolist()
    not_idx = np.where(y_all == 0)[0].tolist()

    return {
        "dataset_summary": {
            "total_samples": len(y_all),
            "train_samples": len(y_tr),
            "dev_samples": len(y_dv),
            "feature_dims": {"audio": AUDIO_DIM, "image": IMAGE_DIM, "text": TEXT_DIM, "total": AUDIO_DIM+IMAGE_DIM+TEXT_DIM},
        },
        "class_distribution": {
            "labels": ["Non-Depressed", "Depressed"],
            "train_counts": [int((y_tr==0).sum()), int((y_tr==1).sum())],
            "dev_counts": [int((y_dv==0).sum()), int((y_dv==1).sum())],
        },
        "pca": {
            "x_dep": [float(pca_coords[i,0]) for i in dep_idx],
            "y_dep": [float(pca_coords[i,1]) for i in dep_idx],
            "x_not": [float(pca_coords[i,0]) for i in not_idx],
            "y_not": [float(pca_coords[i,1]) for i in not_idx]
        }
    }

def get_regression_results() -> dict:
    if not os.path.exists(REG_COMP_CSV):
        return {"error": "Regression results not available. Run training first."}
    
    df = pd.read_csv(REG_COMP_CSV)
    models = {}
    for _, row in df.iterrows():
        models[row['model']] = {
            "mae": round(row['MAE'], 4),
            "rmse": round(row['RMSE'], 4),
            "r2": round(row['R2'], 4)
        }
    
    best_name = df.iloc[df['R2'].idxmax()]['model']
    return {"models": models, "best_model": best_name}

def get_classification_results() -> dict:
    if not os.path.exists(CLF_COMP_CSV):
        return {"error": "Classification results not available. Run training first."}
    
    df = pd.read_csv(CLF_COMP_CSV)
    models = {}
    for _, row in df.iterrows():
        models[row['model']] = {
            "accuracy": round(row['Accuracy'], 4),
            "f1": round(row['F1'], 4)
        }
    
    best_name = df.iloc[df['F1'].idxmax()]['model']
    return {"models": models, "best_model": best_name}

def get_best_models() -> dict:
    reg = get_regression_results()
    clf = get_classification_results()
    return {
        "best_regression": reg.get("best_model", "N/A"),
        "best_classification": clf.get("best_model", "N/A")
    }
