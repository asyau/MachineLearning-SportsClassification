# MachineLearning-SportsClassification

A machine learning project for classifying 100 different sports from images.

## Dataset

The dataset contains **14,492 images** across **100 sports categories**, organized into:

- **Training set**: 11,551 images
- **Validation set**: 1,405 images  
- **Test set**: 1,536 images

### Sports Categories (100 total)

The dataset includes diverse sports such as: air hockey, ampute football, archery, arm wrestling, axe throwing, balance beam, baseball, basketball, BMX, boxing, cricket, curling, fencing, figure skating, football, golf, gymnastics, hockey, judo, lacrosse, NASCAR racing, Olympic wrestling, pole vault, rock climbing, rugby, skiing, snowboarding, surfing, swimming, tennis, volleyball, weightlifting, and many more.

### Dataset Structure

```
MachineLearning-SportsClassification/
├── train/          # Training images (11,551 images)
│   ├── air hockey/
│   ├── archery/
│   ├── baseball/
│   └── ... (100 categories)
├── test/           # Test images (1,536 images)
│   └── ... (100 categories)
└── valid/          # Validation images (1,405 images)
    └── ... (100 categories)
```

## Dataset Download

**Note**: The image datasets are not included in this repository due to size constraints.

### Option 1: Google Drive (Recommended)
Download the dataset from: [Add your link here]

### Option 2: Other sources
[Add alternative download links]

## Installation

```bash
# Clone the repository
git clone https://github.com/asyau/MachineLearning-SportsClassification.git
cd MachineLearning-SportsClassification

# Install dependencies (after you add requirements.txt)
pip install -r requirements.txt

# Download and extract the dataset to the project root
# Ensure you have train/, test/, and valid/ directories
```

## Installation

```bash
# Clone the repository
git clone https://github.com/asyau/MachineLearning-SportsClassification.git
cd MachineLearning-SportsClassification

# Install dependencies
pip install -r requirements.txt

# Download and extract the dataset to the project root
# Ensure you have train/, test/, and valid/ directories
```

## Usage

### Training

Modeli eğitmek için:

```bash
python train.py
```

Training ayarlarını değiştirmek için `config.py` dosyasını düzenleyin:

- **Model Architecture**: `MODEL_NAME` (resnet50, efficientnet_b0-b4, vit_b_16)
- **Batch Size**: `BATCH_SIZE`
- **Learning Rate**: `LEARNING_RATE`
- **Number of Epochs**: `NUM_EPOCHS`
- **Data Augmentation**: `USE_AUGMENTATION`, `AUG_PROB`
- **Class Weights**: `USE_CLASS_WEIGHTS` (imbalance için)
- **Mixed Precision**: `USE_MIXED_PRECISION` (hız için)
- **Early Stopping**: `EARLY_STOPPING`, `EARLY_STOPPING_PATIENCE`

### Evaluation

Eğitilmiş modeli değerlendirmek için:

```bash
# Test set üzerinde değerlendirme
python evaluate.py

# Validation set üzerinde değerlendirme
python evaluate.py --split valid

# Özel checkpoint kullan
python evaluate.py --checkpoint checkpoints/last_model.pt
```

### Inference

Yeni görüntüler için tahmin yapmak için:

```bash
# Tek görüntü
python inference.py --image path/to/image.jpg

# Klasördeki tüm görüntüler
python inference.py --dir path/to/images/

# Top-k tahmin sayısını belirle
python inference.py --image path/to/image.jpg --top_k 10
```

## Project Structure

```
MachineLearning-SportsClassification/
├── README.md
├── requirements.txt
├── config.py              # Yapılandırma dosyası
├── dataset.py             # Dataset ve DataLoader sınıfları
├── model.py               # CNN model tanımları
├── utils.py               # Yardımcı fonksiyonlar
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── inference.py           # Inference script
├── train/                 # Training dataset
├── valid/                 # Validation dataset
├── test/                  # Test dataset
├── checkpoints/           # Model checkpoints (oluşturulur)
└── logs/                  # TensorBoard logs (oluşturulur)
```

## Features

### Model Architecture
- **Transfer Learning**: ResNet50, ResNet101, EfficientNet (B0-B4), Vision Transformer
- **Custom Classifier**: Dropout ve batch normalization ile optimize edilmiş
- **Flexible Backbone**: Pretrained weights ile başlatma veya sıfırdan eğitim

### Training Features
- **Data Augmentation**: Rotation, flipping, color jittering, random erasing, vb.
- **Class Weights**: Imbalanced dataset için otomatik ağırlık hesaplama
- **Learning Rate Scheduling**: Step, Cosine, Plateau, Warmup+Cosine
- **Mixed Precision Training**: FP16 ile hızlı eğitim
- **Gradient Clipping**: Gradient explosion önleme
- **Early Stopping**: Overfitting önleme
- **Model Checkpointing**: En iyi ve son model kaydetme

### Evaluation Metrics
- Accuracy (Top-1, Top-3, Top-5)
- Precision, Recall, F1-Score (Macro ve Weighted)
- Per-class metrics
- Confusion Matrix
- Classification Report

### Visualization
- Training history plots
- Confusion matrix (normalized ve absolute)
- TensorBoard integration

## Configuration

Tüm hyperparameterlar `config.py` dosyasında tanımlıdır. Önemli ayarlar:

```python
MODEL_NAME = 'efficientnet_b3'  # Model seçimi
BATCH_SIZE = 32                  # Batch size
LEARNING_RATE = 0.001            # Learning rate
NUM_EPOCHS = 100                 # Epoch sayısı
USE_CLASS_WEIGHTS = True         # Class imbalance için
USE_MIXED_PRECISION = True       # Hızlı training
EARLY_STOPPING = True            # Early stopping
```

## Results

Model eğitimi tamamlandıktan sonra:
- `checkpoints/best_model.pt`: En iyi model
- `checkpoints/last_model.pt`: Son epoch modeli
- `checkpoints/training_history.png`: Training grafikleri
- `checkpoints/confusion_matrix_test.png`: Confusion matrix
- `checkpoints/metrics_report_test.json`: Detaylı metrikler
- `logs/`: TensorBoard logları

## TensorBoard

Training sırasında metrikleri görselleştirmek için:

```bash
tensorboard --logdir logs/
```

Tarayıcıda `http://localhost:6006` adresini açın.

## License

[Add your license here]
