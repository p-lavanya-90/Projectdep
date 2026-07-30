from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
EXTERNAL_DIR = BASE_DIR / "external_training_data"
IMAGE_ROOT = EXTERNAL_DIR / "DepVidMood" / "Depression Data" / "data"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "DepVidMood_CNN_visual_distress.pth"
SUMMARY_PATH = MODELS_DIR / "depvidmood_cnn_visual_distress_comparison.csv"
SEED = 42
BATCH_SIZE = 64
EPOCHS = 8

DISTRESS_EMOTIONS = {"Angry", "Disgust", "Fear", "Sad"}
NON_DISTRESS_EMOTIONS = {"Happy", "Neutral", "Surprize"}


class VisualDistressCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64 * 6 * 6, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def label_for_emotion(emotion: str) -> int:
    if emotion in DISTRESS_EMOTIONS:
        return 1
    if emotion in NON_DISTRESS_EMOTIONS:
        return 0
    raise ValueError(f"Unknown emotion folder: {emotion}")


class EmotionImageDataset(Dataset):
    def __init__(self, split: str) -> None:
        self.samples = []
        split_dir = IMAGE_ROOT / split
        for emotion_dir in sorted(split_dir.iterdir()):
            if not emotion_dir.is_dir():
                continue
            label = label_for_emotion(emotion_dir.name)
            for path in sorted(emotion_dir.glob("*.png")):
                self.samples.append((path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        image = Image.open(path).convert("L").resize((48, 48))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor, torch.tensor(label, dtype=torch.float32)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    labels = []
    probs = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            prob = torch.sigmoid(logits).cpu().numpy()
            probs.extend(prob.tolist())
            labels.extend(y.numpy().astype(int).tolist())
    y_true = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probs)
    pred = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "AUC": roc_auc_score(y_true, probabilities),
        "MCC": matthews_corrcoef(y_true, pred),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = EmotionImageDataset("train")
    val_ds = EmotionImageDataset("val")
    test_ds = EmotionImageDataset("test")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = VisualDistressCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x.to(device))
            loss = loss_fn(logits, y.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch} loss={np.mean(losses):.4f} "
            f"val_acc={val_metrics['Accuracy']:.4f} val_f1={val_metrics['F1']:.4f}"
        )
        if val_metrics["F1"] > best_val_f1:
            best_val_f1 = val_metrics["F1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "image_size": 48,
            "threshold": 0.5,
            "distress_emotions": sorted(DISTRESS_EMOTIONS),
            "non_distress_emotions": sorted(NON_DISTRESS_EMOTIONS),
        },
        MODEL_PATH,
    )

    rows = []
    for split, loader, ds in [("val", val_loader, val_ds), ("test", test_loader, test_ds)]:
        row = {"split": split, "rows": len(ds), **evaluate(model, loader, device)}
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(SUMMARY_PATH, index=False)
    print(result.round(4).to_string(index=False))
    print(f"Wrote {MODEL_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
