"""
Custom Dataset ve DataLoader sınıfları
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from collections import Counter
from .config import Config


class SportsDataset(Dataset):
    """Sports classification dataset"""
    
    def __init__(self, data_dir, transform=None, is_train=False):
        """
        Args:
            data_dir: Veri dizini (train/valid/test)
            transform: Image transformations
            is_train: Training modunda mı (augmentation için)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_train = is_train
        
        # Tüm görüntüleri ve etiketlerini topla
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        self._load_data()
    
    def _load_data(self):
        """Veriyi yükle ve etiketle"""
        # Klasör isimlerini sıralı olarak al (class names)
        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        # Class to index mapping oluştur
        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
        
        # Tüm görüntüleri topla
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            class_idx = self.class_to_idx[class_name]
            
            # Klasördeki tüm görüntüleri bul
            for img_path in class_dir.iterdir():
                if img_path.suffix in image_extensions:
                    self.images.append(str(img_path))
                    self.labels.append(class_idx)
        
        print(f"Loaded {len(self.images)} images from {len(self.class_to_idx)} classes")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """Bir örnek döndür"""
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Görüntüyü yükle
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Hata durumunda siyah bir görüntü döndür
            image = Image.new('RGB', (Config.IMAGE_SIZE, Config.IMAGE_SIZE), (0, 0, 0))
        
        # Transform uygula
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_counts(self):
        """Her sınıfın örnek sayısını döndür"""
        return Counter(self.labels)


def get_transforms(image_size=224, is_train=False, aug_prob=0.5):
    """
    Data augmentation ve preprocessing transforms
    
    Args:
        image_size: Görüntü boyutu
        is_train: Training modunda mı
        aug_prob: Augmentation uygulanma olasılığı
    """
    if is_train and Config.USE_AUGMENTATION:
        # Training için güçlü augmentation (geliştirilmiş)
        transform = transforms.Compose([
            transforms.Resize((image_size + 48, image_size + 48)),  # Daha büyük resize (32 -> 48)
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=aug_prob),
            transforms.RandomVerticalFlip(p=aug_prob * 0.3),  # Vertical flip daha az
            transforms.RandomRotation(degrees=20),  # Rotation artırıldı (15 -> 20)
            transforms.ColorJitter(
                brightness=0.3,  # Artırıldı (0.2 -> 0.3)
                contrast=0.3,    # Artırıldı (0.2 -> 0.3)
                saturation=0.3,  # Artırıldı (0.2 -> 0.3)
                hue=0.15         # Artırıldı (0.1 -> 0.15)
            ),
            transforms.RandomAffine(
                degrees=10,      # Rotation eklendi
                translate=(0.15, 0.15),  # Artırıldı (0.1 -> 0.15)
                scale=(0.85, 1.15),     # Artırıldı (0.9-1.1 -> 0.85-1.15)
                shear=10         # Artırıldı (5 -> 10)
            ),
            transforms.RandomPerspective(distortion_scale=0.3, p=aug_prob * 0.5),  # Artırıldı (0.2 -> 0.3)
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=aug_prob * 0.4, scale=(0.02, 0.4), ratio=(0.3, 3.3)),  # Güçlendirildi
        ])
    else:
        # Validation/Test için sadece preprocessing
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return transform


def get_class_weights(dataset):
    """
    Class imbalance için ağırlıkları hesapla
    
    Args:
        dataset: SportsDataset instance
    
    Returns:
        torch.Tensor: Her sınıf için ağırlık
    """
    class_counts = dataset.get_class_counts()
    total_samples = len(dataset)
    num_classes = len(class_counts)
    
    # Her sınıfın ağırlığını hesapla (inverse frequency)
    weights = torch.zeros(num_classes)
    for class_idx, count in class_counts.items():
        if count > 0:
            weights[class_idx] = total_samples / (num_classes * count)
        else:
            weights[class_idx] = 0.0
    
    # Normalize et
    weights = weights / weights.sum() * num_classes
    
    return weights


def get_data_loaders(config):
    """
    Train, validation ve test data loader'ları oluştur
    
    Args:
        config: Config instance
    
    Returns:
        train_loader, valid_loader, test_loader, class_weights
    """
    # Transforms
    train_transform = get_transforms(
        image_size=config.IMAGE_SIZE,
        is_train=True,
        aug_prob=config.AUG_PROB
    )
    valid_transform = get_transforms(
        image_size=config.IMAGE_SIZE,
        is_train=False
    )
    test_transform = get_transforms(
        image_size=config.IMAGE_SIZE,
        is_train=False
    )
    
    # Datasets
    train_dataset = SportsDataset(
        data_dir=config.TRAIN_DIR,
        transform=train_transform,
        is_train=True
    )
    valid_dataset = SportsDataset(
        data_dir=config.VALID_DIR,
        transform=valid_transform,
        is_train=False
    )
    test_dataset = SportsDataset(
        data_dir=config.TEST_DIR,
        transform=test_transform,
        is_train=False
    )
    
    # Class weights hesapla
    class_weights = None
    if config.USE_CLASS_WEIGHTS:
        class_weights = get_class_weights(train_dataset)
        print(f"Class weights calculated (min: {class_weights.min():.3f}, max: {class_weights.max():.3f})")
    
    # Weighted sampler (opsiyonel - class weights ile birlikte kullanılabilir)
    weighted_sampler = None
    # Not: Weighted sampler ve class weights birlikte kullanmak overkill olabilir
    # Genelde sadece birini kullanmak yeterli
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True if weighted_sampler is None else False,
        sampler=weighted_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True  # Son incomplete batch'i at
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Class names'i kaydet (evaluation için)
    config.class_names = train_dataset.idx_to_class
    config.num_classes = len(train_dataset.class_to_idx)
    
    return train_loader, valid_loader, test_loader, class_weights


if __name__ == "__main__":
    # Test
    from .config import Config
    config = Config()
    train_loader, valid_loader, test_loader, class_weights = get_data_loaders(config)
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Valid batches: {len(valid_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Bir batch örneği
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels: {labels[:5]}")

