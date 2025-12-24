# Machine Learning Sports Classification

A comprehensive machine learning project for classifying 100 sports categories from images using both deep learning (CNNs) and traditional machine learning approaches.

---

## Overview

This project implements and compares multiple approaches for sports image classification:

- Deep Learning (CNNs): ResNet, EfficientNet, Vision Transformer
- Traditional ML: KNN, Random Forest, SVM
- Feature Extraction: Color Histograms, HOG, LBP
- Dataset Analysis & Augmentation utilities

Dataset size: 14,899 images across 100 sports categories.

---

## Dataset

### Dataset Statistics

- Training: 11,958 images
- Validation: 1,405 images
- Test: 1,536 images
- Total: 14,899 images

### Directory Structure

```
MachineLearning-SportsClassification/
├── train/
│   ├── air hockey/
│   ├── archery/
│   └── ... (100 classes)
├── valid/
│   └── ... (100 classes)
└── test/
    └── ... (100 classes)
```

---

## Quick Start

### 1. Environment Setup

**Important:** Use Python 3.10–3.12 (Python 3.13 is not supported by some ML libraries).

```bash
git clone https://github.com/asyau/MachineLearning-SportsClassification.git
cd MachineLearning-SportsClassification

python -m venv venv
source venv/bin/activate  # macOS / Linux
# venv\Scripts\activate  # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. Dataset Preparation

Ensure your dataset is placed in the project root:

```
train/
valid/
test/
```

---

## Dataset Analysis & Preparation (Optional)

### Analyze Class Distribution

```bash
python analyze_data_distribution.py
```

**Outputs:**

- Class distribution plots
- Imbalance analysis
- CSV summaries
  (saved to `class-dist-analysis/`)

---

### Get Class Sizes

```bash
python get_train_class_sizes.py
```

**Outputs:**

- `train_class_sizes.txt`

---

### Augment Small Classes

Augments classes with fewer than 110 images.

```bash
python augment_small_classes.py
```

---

### Create Train/Validation/Test Split

```bash
python create_train_val_test_split.py
```

Edit the script to configure:

- Input directory
- Output directory
- Split ratios (default: 80 / 10 / 10)

---

## Deep Learning (CNN) – Recommended

### Training

```bash
python train.py
```

or

```bash
cd cnn
python train.py
```

### Configuration (`cnn/config.py`)

```python
MODEL_NAME = 'efficientnet_b4'  # resnet50, efficientnet_b0-b4, vit_b_16
BATCH_SIZE = 24
NUM_EPOCHS = 15
LEARNING_RATE = 0.0005

USE_CLASS_WEIGHTS = True
USE_MIXED_PRECISION = True
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 5
```

---

### Evaluation

```bash
python evaluate.py
```

**Supported metrics:**

- Top-1 / Top-3 / Top-5 Accuracy
- Precision, Recall, F1 (Macro & Weighted)
- Confusion Matrix
- Per-class metrics

---

### Inference

**Single image:**

```bash
python inference.py --image path/to/image.jpg
```

**Directory:**

```bash
python inference.py --dir path/to/images/
```

**Top-K predictions:**

```bash
python inference.py --image path/to/image.jpg --top_k 10
```

**Custom checkpoint:**

```bash
python inference.py --image path/to/image.jpg --checkpoint checkpoints/best_model.pt
```

---

### TensorBoard

```bash
tensorboard --logdir logs/
```

Open: http://localhost:6006

---

## Traditional Machine Learning Approaches

### K-Nearest Neighbors (KNN)

```bash
python knn_color_histogram_only.py
python knn_hog_only.py
python knn_lbp_only.py
```

---

### Random Forest (RF)

```bash
python rf_color_histogram_only.py
python rf_hog_only.py
python rf_lbp_only.py
```

---

### Support Vector Machine (SVM)

Edit `SVM_hog_color_lbp.py`:

```python
BASE_DIR = "."
SELECTED_FEATURE = 'color'  # 'color', 'hog', 'lbp'
USE_PCA = True
```

**Run:**

```bash
python SVM_hog_color_lbp.py
```

Outputs saved to `svm_output/`.

---

### Combined Features

```bash
python concatinated.py
```

Uses Color + HOG + LBP together.

---

## Project Structure

```
MachineLearning-SportsClassification/
├── cnn/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils.py
│
├── train.py
├── evaluate.py
├── inference.py
│
├── analyze_data_distribution.py
├── augment_small_classes.py
├── create_train_val_test_split.py
├── get_train_class_sizes.py
│
├── knn_*.py
├── rf_*.py
├── SVM_hog_color_lbp.py
├── concatinated.py
│
├── train/
├── valid/
├── test/
├── checkpoints/
├── logs/
└── class-dist-analysis/
```

---

## Requirements (Core)

Key dependencies (see `requirements.txt` for full list):

- torch
- torchvision
- tensorflow
- scikit-learn
- opencv-python
- numpy
- pandas
- matplotlib
- seaborn
- tensorboard

---

## Troubleshooting

### CUDA Out of Memory

- Reduce `BATCH_SIZE`
- Reduce `IMAGE_SIZE`
- Enable mixed precision

### Slow Training

- Use smaller model (`efficientnet_b0`)
- Enable mixed precision
- Increase `NUM_WORKERS`

### Dataset Not Found

- Ensure `train/`, `valid/`, `test/` exist in project root
- Check `BASE_DIR` in scripts

---

## Notes

- CNN models give significantly better accuracy than traditional ML
- Traditional ML is useful for baseline comparisons
- Class imbalance handling is strongly recommended

---
