import os
import copy
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


# =====================
# FORCE GPU 0
# =====================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your GPU environment.")

DEVICE = torch.device("cuda:0")


# =====================
# CONFIG
# =====================
DATASET_ROOT = "/home/feg48/hw/Dataset"   
MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
BATCH_SIZE = 16
EPOCHS = 15
NUM_WORKERS = 4
PATIENCE = 5
SEED = 42
OUTPUT_DIR = "./dinov2_outputs"
RESULT_CSV = os.path.join(OUTPUT_DIR, "all_results.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================
# SEED
# =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# =====================
# DATASET
# Folder format:
# DATASET_ROOT/
#   Knee_OA/*.png
#   NoKnee_OA/*.png
# =====================
class KneeDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def collect_samples(root_dir):
    root = Path(root_dir)
    samples = []

    class_map = {
        "NoKnee_OA": 0,
        "Knee_OA": 1,
    }

    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    for class_name, label in class_map.items():
        class_dir = root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing folder: {class_dir}")

        for p in class_dir.rglob("*"):
            if p.suffix.lower() in valid_ext:
                samples.append((str(p), label))

    if len(samples) == 0:
        raise ValueError("No images found in dataset folders.")

    return samples


def split_data(samples, train_percent, seed=42):
    paths = [x[0] for x in samples]
    labels = [x[1] for x in samples]

    # 80/20 train+val/test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        paths,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=seed,
    )

    # 80/20 train/val from trainval
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.2,
        stratify=y_trainval,
        random_state=seed,
    )

    # apply 20%, 50%, 100% only to training set
    if train_percent < 1.0:
        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=train_percent,
            stratify=y_train,
            random_state=seed,
        )

    train_samples = list(zip(X_train, y_train))
    val_samples = list(zip(X_val, y_val))
    test_samples = list(zip(X_test, y_test))

    return train_samples, val_samples, test_samples


# =====================
# MODEL
# =====================
class DINOv2Classifier(nn.Module):
    def __init__(self, model_name, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits


# =====================
# METRICS
# =====================
def compute_metrics(y_true, y_pred, y_prob):
    out = {}
    out["accuracy"] = accuracy_score(y_true, y_pred)
    out["precision"] = precision_score(y_true, y_pred, zero_division=0)
    out["recall"] = recall_score(y_true, y_pred, zero_division=0)
    out["f1"] = f1_score(y_true, y_pred, zero_division=0)
    out["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        out["roc_auc"] = float("nan")

    return out


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    y_true_all, y_pred_all, y_prob_all = [], [], []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        y_true_all.extend(labels.detach().cpu().numpy().tolist())
        y_pred_all.extend(preds.detach().cpu().numpy().tolist())
        y_prob_all.extend(probs.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(y_true_all, y_pred_all, y_prob_all)
    metrics["loss"] = avg_loss
    return metrics


# =====================
# EXPERIMENT
# =====================
def run_experiment(train_percent, freeze_backbone):
    print(f"\n=== Running: {train_percent*100:.1f}% | Freeze={freeze_backbone} ===")

    samples = collect_samples(DATASET_ROOT)
    train_samples, val_samples, test_samples = split_data(samples, train_percent, seed=SEED)

    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    train_labels = [x[1] for x in train_samples]
    n0 = sum(1 for x in train_labels if x == 0)
    n1 = sum(1 for x in train_labels if x == 1)
    print(f"Train class counts -> NoKnee_OA: {n0}, Knee_OA: {n1}")

    # transforms matched to the model
    temp_model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    data_cfg = resolve_data_config({}, model=temp_model)
    train_tf = create_transform(**data_cfg, is_training=True)
    eval_tf = create_transform(**data_cfg, is_training=False)

    train_ds = KneeDataset(train_samples, transform=train_tf)
    val_ds = KneeDataset(val_samples, transform=eval_tf)
    test_ds = KneeDataset(test_samples, transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = DINOv2Classifier(MODEL_NAME, dropout=0.3).to(DEVICE)

    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    # class weights for imbalance
    class_weights = torch.tensor(
        [1.0 / max(n0, 1), 1.0 / max(n1, 1)],
        dtype=torch.float32,
        device=DEVICE,
    )
    class_weights = class_weights / class_weights.sum() * 2.0

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if freeze_backbone:
        optimizer = torch.optim.AdamW(
            model.head.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": 1e-5},
                {"params": model.head.parameters(), "lr": 1e-3},
            ],
            weight_decay=1e-4,
        )

    best_model = None
    best_score = -1.0
    bad_epochs = 0

    for epoch in range(EPOCHS):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer=None)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss {train_metrics['loss']:.4f} | "
            f"Train F1 {train_metrics['f1']:.4f} | "
            f"Val Loss {val_metrics['loss']:.4f} | "
            f"Val F1 {val_metrics['f1']:.4f} | "
            f"Val Macro-F1 {val_metrics['macro_f1']:.4f} | "
            f"Val AUC {val_metrics['roc_auc']:.4f}"
        )

        score = val_metrics["macro_f1"]
        if score > best_score:
            best_score = score
            best_model = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("Early stopping.")
                break

    model.load_state_dict(best_model)
    test_metrics = run_epoch(model, test_loader, criterion, optimizer=None)

    print("Test Results:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    return test_metrics


# =====================
# MAIN: RUN 6 EXPERIMENTS
# =====================
def main():
    print("Using device:", DEVICE)
    print("GPU name:", torch.cuda.get_device_name(0))

    results = []

    for train_percent in [0.2, 0.5, 1.0]:
        for freeze_backbone in [True, False]:
            metrics = run_experiment(train_percent, freeze_backbone)
            row = {
                "data_percent": train_percent,
                "freeze_backbone": freeze_backbone,
                **metrics,
            }
            results.append(row)

            pd.DataFrame(results).to_csv(RESULT_CSV, index=False)

    df = pd.DataFrame(results)
    print("\n===== Final Results =====")
    print(df)
    df.to_csv(RESULT_CSV, index=False)
    print(f"\nSaved results to: {RESULT_CSV}")


if __name__ == "__main__":
    main()