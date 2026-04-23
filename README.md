# Knee OA Classification - Final Project
Adaptation of Vision Foundation Model for Knee Osteoarthritis Classification

---

## Introduction
This project investigates the use of pretrained vision foundation model for automated detection of knee osteoarthritis (OA) from anterior-posterior (AP) radiographs. Specifically, we use a DINOv2 (**EDIT if we chose different model) vision transformer to classify images into OA and non-OA categories.

A key focus of this work is evaluating how model performance changes under different data availability conditions, reflecting real-world clinical scenarios where labeled medical data is often limited.

---

## Methods

### Dataset
The dataset consists of 4,156 AP knee radiographs from 2,655 unique subjects. Each image is labeled using the Kellgren–Lawrence (KL) grading system:

- KL < 2 → No OA
- KL ≥ 2 → OA

Images are organized into two classes:
- **Knee OA**
- **No-Knee OA**

Images are labeled as follows:
- 0: No OA (KL < 2)
- 1: OA (KL ≥ 2)

Some subjects have one radiograph (left or right knee), while others have both. To prevent data leakage, all dataset splits (train/validation/test) are performed at the subject level, ensuring that images from the same patient do not appear in multiple splits.

---

### Preprocessing
- Images resized to 224 × 224
- Normalized using (**EDIT???)
- ROI extraction (**REMOVE if we don't do)

---

### Model
- Backbone: DINOv2 (ViT) (**EDIT if we choose a different model)
- Classifier: MLP (Linear → ReLU → Dropout → Linear)

---

### Training
- Loss function: Cross-Entropy Loss
- Optimizer: AdamW
- Batch size: 16
- Epochs: 10-20
- Early stopping based on validation performance

---

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Macro-F1
- ROC-AUC

---

## Experimental Setup
We evaluate model performance under three data availability settings:

- 20% training data (low-data availability)
- 50% training data (moderate-data availability)
- 100% training data (full-data availability)

All experiments use the same architecture and training settings to ensure fair comparison across conditions.

---

## Results (TO BE ADDED)

### Overall performance table:
| Data % | Accuracy | Precision | Recall | F1-score | Macro-F1 | ROC-AUC |
|--------|----------|-----------|--------|----------|----------|---------|
| 20%    |          |           |        |          |          |         |
| 50%    |          |           |        |          |          |         |
| 100%   |          |           |        |          |          |         |

### Figures (TO BE ADDED)
- ROC curves for each data regime??  
- Training vs validation loss curves??  
- Confusion matrices??

---

## Discussion (TO BE ADDED)

1. **Data Efficiency**
- Effect of dataset size on performance (20% → 100%)

2. **Model Behavior**
- Model behavior under low-data vs full-data availability

3. **Adaptiation Strategy**
- Effectiveness of pretrained DINOv2 (**EDIT if other model(s) used) feature extraction

4. **Stability**
- Stability and consistency across training runs and data availability 

---

## Conclusion (TO BE ADDED)
This project investigates the use of vision foundation models for automated knee osteoarthritis detection and evaluates how data availability impacts classification performance in a medical imaging setting.
