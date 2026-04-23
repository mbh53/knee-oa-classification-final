# Knee OA Classification - Final Project
Adaptation of Vision Foundation Model for Knee Osteoarthritis Classification

## Overview
This project implements a binary classification model to detect Knee Osteoarthritis (OA) from radiographs using a pretrained DINOv2 (**edit if we use a different model) vision transformer.

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
