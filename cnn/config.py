"""
Configuration file for Sports Classification CNN Model
Tüm hyperparameterlar ve yapılandırma ayarları burada tanımlanır.
"""

import os
from pathlib import Path

class Config:
    """Model ve training yapılandırması"""
    
    # Dataset paths - parent directory'ye çık (cnn klasöründen bir üst)
    BASE_DIR = Path(__file__).parent.parent
    TRAIN_DIR = BASE_DIR / 'train'
    VALID_DIR = BASE_DIR / 'valid'
    TEST_DIR = BASE_DIR / 'test'
    
    # Model settings
    NUM_CLASSES = 100
    IMAGE_SIZE = 256  # Daha büyük image size (224 -> 256) accuracy artırır
    BATCH_SIZE = 24  # Image size artınca batch size düşürüldü (32 -> 24) (GPU memory için)
    NUM_WORKERS = 2  # DataLoader için (sistem önerisi: 2)
    PIN_MEMORY = True
    
    # Model architecture
    MODEL_NAME = 'efficientnet_b4'  # B3 -> B4 (daha büyük model, daha iyi accuracy)
    PRETRAINED = True
    FREEZE_BACKBONE = False  # Transfer learning için backbone'u dondur
    
    # Training hyperparameters
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0005  # Daha küçük LR (0.001 -> 0.0005) daha stabil training
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    
    # Learning rate scheduling
    LR_SCHEDULER = 'warmup_cosine'  # Warmup + Cosine (daha iyi convergence)
    LR_STEP_SIZE = 30  # StepLR için
    LR_GAMMA = 0.1  # StepLR için
    LR_MIN = 1e-6  # Minimum learning rate
    WARMUP_EPOCHS = 10  # Warmup epochs artırıldı (5 -> 10)
    
    # Optimizer
    OPTIMIZER = 'adamw'  # 'adam', 'adamw', 'sgd'
    
    # Loss function
    USE_CLASS_WEIGHTS = True  # Imbalance için class weights kullan
    LABEL_SMOOTHING = 0.1  # Label smoothing factor (0 = disabled)
    USE_FOCAL_LOSS = False  # Focal Loss (imbalance için, class weights ile birlikte kullanılabilir)
    FOCAL_ALPHA = 0.25  # Focal Loss alpha
    FOCAL_GAMMA = 2.0  # Focal Loss gamma
    
    # Data augmentation
    USE_AUGMENTATION = True
    AUG_PROB = 0.7  # Augmentation olasılığı artırıldı (0.5 -> 0.7)
    USE_MIXUP = True  # Mixup augmentation (accuracy artırır)
    MIXUP_ALPHA = 0.2  # Mixup alpha değeri
    USE_CUTMIX = True  # CutMix augmentation
    CUTMIX_ALPHA = 1.0  # CutMix alpha değeri
    
    # Training techniques
    USE_MIXED_PRECISION = True  # FP16 training (hız için)
    GRADIENT_CLIP = 1.0  # Gradient clipping (0 = disabled)
    ACCUMULATION_STEPS = 1  # Gradient accumulation
    
    # Early stopping
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 20  # Patience artırıldı (15 -> 20)
    EARLY_STOPPING_MIN_DELTA = 0.0005  # Min delta azaltıldı (daha hassas)
    
    # Model checkpointing
    SAVE_DIR = BASE_DIR / 'checkpoints'
    SAVE_BEST_ONLY = True  # Sadece en iyi modeli kaydet
    SAVE_LAST = True  # Son epoch'u da kaydet
    
    # Evaluation
    EVAL_METRICS = ['accuracy', 'precision', 'recall', 'f1']
    TOP_K_ACCURACY = [1, 3, 5]  # Top-k accuracy hesapla
    USE_TTA = True  # Test Time Augmentation (inference'da accuracy artırır)
    TTA_NUM = 5  # TTA için augmentation sayısı
    
    # Logging
    LOG_DIR = BASE_DIR / 'logs'
    USE_TENSORBOARD = True
    PRINT_FREQ = 50  # Her N batch'te bir log yazdır
    
    # Reproducibility
    SEED = 42
    DETERMINISTIC = True  # Tam reproducibility için (yavaş olabilir)
    
    # Device
    DEVICE = 'cuda'  # 'cuda' veya 'cpu'
    
    # Inference
    INFERENCE_BATCH_SIZE = 64
    
    @classmethod
    def create_dirs(cls):
        """Gerekli dizinleri oluştur"""
        cls.SAVE_DIR.mkdir(exist_ok=True)
        cls.LOG_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def get_class_weights_path(cls):
        """Class weights dosya yolu"""
        return cls.SAVE_DIR / 'class_weights.pt'
    
    @classmethod
    def get_best_model_path(cls):
        """En iyi model dosya yolu"""
        return cls.SAVE_DIR / 'best_model.pt'
    
    @classmethod
    def get_last_model_path(cls):
        """Son model dosya yolu"""
        return cls.SAVE_DIR / 'last_model.pt'
    
    @classmethod
    def get_config_dict(cls):
        """Yapılandırmayı dictionary olarak döndür"""
        return {
            'model_name': cls.MODEL_NAME,
            'num_classes': cls.NUM_CLASSES,
            'image_size': cls.IMAGE_SIZE,
            'batch_size': cls.BATCH_SIZE,
            'num_epochs': cls.NUM_EPOCHS,
            'learning_rate': cls.LEARNING_RATE,
            'optimizer': cls.OPTIMIZER,
            'lr_scheduler': cls.LR_SCHEDULER,
            'use_class_weights': cls.USE_CLASS_WEIGHTS,
            'label_smoothing': cls.LABEL_SMOOTHING,
            'use_mixed_precision': cls.USE_MIXED_PRECISION,
        }

