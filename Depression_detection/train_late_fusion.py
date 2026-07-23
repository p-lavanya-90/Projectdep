"""Train a learned late-fusion depression classifier on the existing features.

Each modality is encoded independently before a learned gate assigns its
contribution. This is different from simple feature concatenation.
"""

from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, fbeta_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

SEED = 42
AUDIO_DIM, IMAGE_DIM, TEXT_DIM = 153, 160, 768
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = Path(__file__).resolve().parent
DATA = BASE / "final_dataset"
MODELS = BASE / "models"


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


class LateFusionClassifier(nn.Module):
    """Independent modality encoders followed by sample-specific soft gating."""

    def __init__(self, hidden_dim=64, dropout=0.35):
        super().__init__()

        def branch(input_dim):
            return nn.Sequential(
                nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            )

        self.audio_branch = branch(AUDIO_DIM)
        self.image_branch = branch(IMAGE_DIM)
        self.text_branch = branch(TEXT_DIM)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 3), nn.Softmax(dim=1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim, 2),
        )

    def forward(self, audio, image, text):
        encoded = [self.audio_branch(audio), self.image_branch(image), self.text_branch(text)]
        weights = self.gate(torch.cat(encoded, dim=1))
        fused = sum(weights[:, index:index + 1] * value for index, value in enumerate(encoded))
        return self.classifier(fused), weights


def split_modalities(features):
    return (
        features[:, :AUDIO_DIM],
        features[:, AUDIO_DIM:AUDIO_DIM + IMAGE_DIM],
        features[:, AUDIO_DIM + IMAGE_DIM:AUDIO_DIM + IMAGE_DIM + TEXT_DIM],
    )


def choose_threshold(labels, probabilities):
    """Select a balanced development-set threshold using F1, then MCC."""
    best_threshold, best_f1, best_mcc = 0.50, -1.0, -1.0
    for threshold in np.arange(0.05, 0.96, 0.01):
        predictions = probabilities >= threshold
        score_f1 = f1_score(labels, predictions, zero_division=0)
        score_mcc = matthews_corrcoef(labels, predictions)
        if score_f1 > best_f1 or (score_f1 == best_f1 and score_mcc > best_mcc):
            best_threshold, best_f1, best_mcc = float(threshold), float(score_f1), float(score_mcc)
    return best_threshold, best_f1, best_mcc


@torch.no_grad()
def probabilities(model, features):
    model.eval()
    audio, image, text = (torch.tensor(value, dtype=torch.float32, device=DEVICE) for value in split_modalities(features))
    logits, attention = model(audio, image, text)
    return (
        torch.softmax(logits, dim=1)[:, 1].cpu().numpy(),
        attention.cpu().numpy(),
    )


def main():
    set_seed()
    MODELS.mkdir(exist_ok=True)
    x_train = np.load(DATA / "X_train.npy").astype(np.float32)
    x_dev = np.load(DATA / "X_dev.npy").astype(np.float32)
    x_test = np.load(DATA / "X_test.npy").astype(np.float32)
    y_train = np.load(DATA / "y_train_binary.npy").astype(np.int64)
    y_dev = np.load(DATA / "y_dev_binary.npy").astype(np.int64)
    y_test = np.load(DATA / "y_test_binary.npy").astype(np.int64)

    expected_dim = AUDIO_DIM + IMAGE_DIM + TEXT_DIM
    if x_train.shape[1] != expected_dim:
        raise ValueError(f"Expected {expected_dim} features, received {x_train.shape[1]}.")

    train_modalities = [torch.tensor(value, dtype=torch.float32, device=DEVICE) for value in split_modalities(x_train)]
    train_labels = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    positive_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, positive_weight], dtype=torch.float32, device=DEVICE)
    )

    model = LateFusionClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    best_state, best_dev_f1, best_dev_mcc, best_threshold, stale_epochs = None, -1.0, -1.0, 0.50, 0

    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(*train_modalities)
        loss = criterion(logits, train_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        dev_probabilities, _ = probabilities(model, x_dev)
        threshold, dev_f1, dev_mcc = choose_threshold(y_dev, dev_probabilities)
        if dev_f1 > best_dev_f1 or (dev_f1 == best_dev_f1 and dev_mcc > best_dev_mcc):
            best_state, best_dev_f1, best_dev_mcc, best_threshold = (
                deepcopy(model.state_dict()), dev_f1, dev_mcc, threshold
            )
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 40:
            break

    model.load_state_dict(best_state)
    test_probabilities, test_attention = probabilities(model, x_test)
    predictions = (test_probabilities >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, predictions)
    metrics = {
        "model": "Learned Late Fusion",
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, test_probabilities),
        "MCC": matthews_corrcoef(y_test, predictions),
        "Threshold": best_threshold,
        "Dev_F1": best_dev_f1,
        "Dev_MCC": best_dev_mcc,
        "Dev_F2": fbeta_score(y_dev, probabilities(model, x_dev)[0] >= best_threshold, beta=2.0, zero_division=0),
        "Epochs": epoch,
        "TN": cm[0, 0], "FP": cm[0, 1], "FN": cm[1, 0], "TP": cm[1, 1],
        "Mean_Audio_Attention": float(test_attention[:, 0].mean()),
        "Mean_Visual_Attention": float(test_attention[:, 1].mean()),
        "Mean_Text_Attention": float(test_attention[:, 2].mean()),
    }
    pd.DataFrame([metrics]).to_csv(MODELS / "late_fusion_comparison.csv", index=False)
    torch.save(
        {"model_state": model.state_dict(), "threshold": best_threshold, "dimensions": (AUDIO_DIM, IMAGE_DIM, TEXT_DIM)},
        MODELS / "late_fusion_classifier.pth",
    )
    print(pd.DataFrame([metrics]).round(4).to_string(index=False))
    print(f"Saved model: {MODELS / 'late_fusion_classifier.pth'}")


if __name__ == "__main__":
    main()
