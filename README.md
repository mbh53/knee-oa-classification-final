# Knee OA Classification - Final Project
Adaptation of Vision Foundation Model for Knee Osteoarthritis Classification

## Overview
This project implements a binary classification model to detect Knee Osteoarthritis (OA) from radiographs using a pretrained DINOv2 (**edit if we use a different model) vision transformer.

## Dataset
The dataset consists of anterior-posterior (AP) knee radiographs organized into two classes:
- **Knee OA**
- **No-Knee OA**
The dataset contains 4,156 images from 2,655 unique subjects. Each image is labeled based on the Kellgren–Lawrence (KL) grading system:
- KL < 2 → No OA
- KL ≥ 2 → OA

Some subjects have one radiograph (left or right knee), while others have both. To prevent data leakage, all dataset splits (train/validation/test) are performed at the **subject level**, ensuring that images from the same patient do not appear in multiple splits.

## Preprocessing
- Images resized to 224 × 224
- Normalized using ???
- ROI extraction (**remove if we don't do)

## Labels
- 0: No OA (KL < 2)
- 1: OA (KL ≥ 2)

## Model
- Backbone: DINOv2 (ViT) (**edit if we choose a different model)
- Classifier: MLP (Linear → ReLU → Dropout → Linear)

## Training
- Loss: Cross-Entropy (with class weights)
- Optimizer: AdamW
- Batch size: 16
- Epochs: 10
- Early stopping applied

## Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Macro-F1
- ROC-AUC

## Experiments
- 20% data
- 50% data
- 100% data
