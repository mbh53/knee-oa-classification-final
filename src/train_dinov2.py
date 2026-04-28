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
    auc,
    confusion_matrix,
    precision_score,
    precision_recall_curve,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")


# =====================
# CONFIG
# =====================
DATASET_ROOT = str("Dataset")
MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
BATCH_SIZE = 16
EPOCHS = 10
NUM_WORKERS = 4
PATIENCE = 5
SEED = 42
OUTPUT_DIR = "./dinov2_outputs"
RESULT_CSV = os.path.join(OUTPUT_DIR, "all_results.csv")
FIG_ROC = os.path.join(OUTPUT_DIR, "roc_curves.png")
FIG_PR = os.path.join(OUTPUT_DIR, "precision_recall_curves.png")
FIG_DATA_EFFICIENCY = os.path.join(OUTPUT_DIR, "data_efficiency.png")
CM_DIR = os.path.join(OUTPUT_DIR, "confusion_matrices")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)


# =====================
# SEED
# =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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


def collect_predictions(model, loader):
    model.eval()
    y_true_all, y_pred_all, y_prob_all = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=torch.cuda.is_available())
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            y_true_all.extend(labels.numpy().tolist())
            y_pred_all.extend(preds.detach().cpu().numpy().tolist())
            y_prob_all.extend(probs.detach().cpu().numpy().tolist())

    return np.array(y_true_all), np.array(y_pred_all), np.array(y_prob_all)


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

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
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
    test_y_true, test_y_pred, test_y_prob = collect_predictions(model, test_loader)
    try:
        fpr, tpr, _ = roc_curve(test_y_true, test_y_prob)
        precision, recall, _ = precision_recall_curve(test_y_true, test_y_prob)
        test_metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        test_metrics["pr_curve"] = {"precision": precision.tolist(), "recall": recall.tolist()}
        test_metrics["pr_auc"] = auc(recall, precision)
        cm = confusion_matrix(test_y_true, test_y_pred)
        test_metrics["confusion_matrix"] = cm.tolist()
    except ValueError:
        test_metrics["roc_curve"] = None
        test_metrics["pr_curve"] = None
        test_metrics["pr_auc"] = float("nan")
        test_metrics["confusion_matrix"] = None

    print("Test Results:")
    for k, v in test_metrics.items():
        if k in {"roc_curve", "pr_curve", "confusion_matrix"}:
            continue
        print(f"  {k}: {v:.4f}")

    return test_metrics


def save_confusion_matrices(results_df):
    for idx, row in results_df.iterrows():
        cm_data = row.get("confusion_matrix")
        if isinstance(cm_data, list):
            cm = np.array(cm_data)
            data_pct = int(row["data_percent"] * 100)
            strategy = "frozen" if row["freeze_backbone"] else "finetuned"
            fig, ax = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NoKnee_OA", "Knee_OA"])
            disp.plot(ax=ax, cmap="Blues")
            plt.title(f"Confusion Matrix ({data_pct}% | {strategy})")
            plt.tight_layout()
            filename = os.path.join(CM_DIR, f"cm_{data_pct}pct_{strategy}.png")
            plt.savefig(filename, dpi=200)
            plt.close()


def save_curves(results_df):
    plot_df = results_df.copy()
    plot_df["strategy"] = plot_df["freeze_backbone"].map({True: "Frozen backbone", False: "Fine-tuned"})
    save_confusion_matrices(results_df)

    plt.figure(figsize=(7, 6))
    for _, row in plot_df.iterrows():
        roc_data = row.get("roc_curve")
        if isinstance(roc_data, dict):
            label = f"{int(row['data_percent'] * 100)}% | {row['strategy']} (AUC={row['roc_auc']:.3f})"
            plt.plot(roc_data["fpr"], roc_data["tpr"], linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_ROC, dpi=200)
    plt.close()

    plt.figure(figsize=(7, 6))
    for _, row in plot_df.iterrows():
        pr_data = row.get("pr_curve")
        if isinstance(pr_data, dict):
            label = f"{int(row['data_percent'] * 100)}% | {row['strategy']} (AP={row['pr_auc']:.3f})"
            plt.plot(pr_data["recall"], pr_data["precision"], linewidth=2, label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_PR, dpi=200)
    plt.close()

    summary = (
        plot_df.groupby("data_percent", as_index=False)
        .agg(accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"), roc_auc=("roc_auc", "mean"))
        .sort_values("data_percent")
    )

    plt.figure(figsize=(7, 6))
    plt.plot(summary["data_percent"] * 100, summary["accuracy"], marker="o", label="Accuracy")
    plt.plot(summary["data_percent"] * 100, summary["macro_f1"], marker="o", label="Macro-F1")
    plt.plot(summary["data_percent"] * 100, summary["roc_auc"], marker="o", label="ROC-AUC")
    plt.xlabel("Training data used (%)")
    plt.ylabel("Score")
    plt.title("Data Efficiency")
    plt.xticks([20, 50, 100])
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DATA_EFFICIENCY, dpi=200)
    plt.close()


# =====================
# MAIN: RUN 6 EXPERIMENTS
# =====================
def main():
    print("Using device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
    else:
        print("GPU name: CPU fallback")

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
    csv_df = df.drop(columns=["roc_curve", "pr_curve", "confusion_matrix"], errors="ignore")
    print("\n===== Final Results =====")
    print(df)
    csv_df.to_csv(RESULT_CSV, index=False)
    save_curves(df)
    print(f"\nSaved results to: {RESULT_CSV}")
    print(f"Saved ROC curves to: {FIG_ROC}")
    print(f"Saved PR curves to: {FIG_PR}")
    print(f"Saved data efficiency plot to: {FIG_DATA_EFFICIENCY}")
    print(f"Saved confusion matrices to: {CM_DIR}")


if __name__ == "__main__":
    main()