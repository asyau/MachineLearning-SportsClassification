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
    
    def __init__(self, data_dir, transform=None, is_train=False, class_sample_counts=None, mean_sample_count=None):
        """
        Args:
            data_dir: Veri dizini (train/valid/test)
            transform: Image transformations
            is_train: Training modunda mı (augmentation için)
            class_sample_counts: Her sınıfın örnek sayısı dict (sınıf bazlı augmentation için)
            mean_sample_count: Ortalama örnek sayısı (sınıf bazlı augmentation için)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_train = is_train
        self.class_sample_counts = class_sample_counts or {}
        self.mean_sample_count = mean_sample_count
        
        # Tüm görüntüleri ve etiketlerini topla
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.image_class_indices = []  # Her görüntünün hangi sınıfa ait olduğunu tut
        
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
                    self.image_class_indices.append(class_idx)
        
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
        
        # Sınıf bazlı transform uygula (eğer training modundaysa ve sınıf bilgisi varsa)
        if self.transform:
            if self.is_train and self.class_sample_counts and self.mean_sample_count:
                # Bu sınıfın örnek sayısını al
                class_name = self.idx_to_class[label]
                class_sample_count = self.class_sample_counts.get(class_name, self.mean_sample_count)
                # Sınıf bazlı transform oluştur
                class_transform = get_transforms(
                    image_size=Config.IMAGE_SIZE,
                    is_train=True,
                    aug_prob=Config.AUG_PROB,
                    class_sample_count=class_sample_count,
                    mean_sample_count=self.mean_sample_count
                )
                image = class_transform(image)
            else:
                # Normal transform
                image = self.transform(image)
        
        return image, label
    
    def get_class_counts(self):
        """Her sınıfın örnek sayısını döndür"""
        return Counter(self.labels)


def get_transforms(image_size=224, is_train=False, aug_prob=0.5, class_sample_count=None, mean_sample_count=None):
    """
    Data augmentation ve preprocessing transforms
    
    Args:
        image_size: Görüntü boyutu
        is_train: Training modunda mı
        aug_prob: Augmentation uygulanma olasılığı
        class_sample_count: Bu sınıfın örnek sayısı (sınıf bazlı augmentation için)
        mean_sample_count: Ortalama örnek sayısı (sınıf bazlı augmentation için)
    """
    if is_train and Config.USE_AUGMENTATION:
        # Sınıf bazlı augmentation: Az örnekli sınıflar için daha agresif augmentation
        if class_sample_count is not None and mean_sample_count is not None:
            # Örnek sayısı ortalamanın altındaysa daha güçlü augmentation
            if class_sample_count < mean_sample_count * 0.8:  # %20'den fazla eksikse
                aug_strength = 1.3  # %30 daha güçlü augmentation
                rotation_degrees = 25  # Daha fazla rotation
                color_jitter_strength = 0.4  # Daha güçlü color jitter
            elif class_sample_count < mean_sample_count * 0.9:  # %10-20 eksikse
                aug_strength = 1.15  # %15 daha güçlü augmentation
                rotation_degrees = 22
                color_jitter_strength = 0.35
            else:
                aug_strength = 1.0  # Normal augmentation
                rotation_degrees = 20
                color_jitter_strength = 0.3
        else:
            # Sınıf bilgisi yoksa normal augmentation
            aug_strength = 1.0
            rotation_degrees = 20
            color_jitter_strength = 0.3
        # Training için güçlü augmentation (sınıf bazlı ayarlanabilir)
        transform = transforms.Compose([
            transforms.Resize((image_size + int(48 * aug_strength), image_size + int(48 * aug_strength))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=min(aug_prob * aug_strength, 1.0)),
            transforms.RandomVerticalFlip(p=min(aug_prob * 0.3 * aug_strength, 1.0)),
            transforms.RandomRotation(degrees=int(rotation_degrees * aug_strength)),
            transforms.ColorJitter(
                brightness=color_jitter_strength * aug_strength,
                contrast=color_jitter_strength * aug_strength,
                saturation=color_jitter_strength * aug_strength,
                hue=0.15 * aug_strength
            ),
            transforms.RandomAffine(
                degrees=int(10 * aug_strength),
                translate=(0.15 * aug_strength, 0.15 * aug_strength),
                scale=(max(0.85 - 0.1 * (aug_strength - 1), 0.75), min(1.15 + 0.1 * (aug_strength - 1), 1.25)),
                shear=int(10 * aug_strength)
            ),
            transforms.RandomPerspective(
                distortion_scale=min(0.3 * aug_strength, 0.5),
                p=min(aug_prob * 0.5 * aug_strength, 1.0)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(
                p=min(aug_prob * 0.4 * aug_strength, 1.0),
                scale=(0.02, min(0.4 * aug_strength, 0.5)),
                ratio=(0.3, 3.3)
            ),
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
    # Önce train dataset'i yükle (sınıf sayılarını almak için)
    temp_dataset = SportsDataset(
        data_dir=config.TRAIN_DIR,
        transform=None,
        is_train=False
    )
    
    # Sınıf bazlı örnek sayılarını hesapla
    class_counts = temp_dataset.get_class_counts()
    class_sample_counts = {}
    for class_idx, count in class_counts.items():
        class_name = temp_dataset.idx_to_class[class_idx]
        class_sample_counts[class_name] = count
    
    mean_sample_count = sum(class_counts.values()) / len(class_counts) if class_counts else None
    
    # Şimdi gerçek train dataset'i oluştur (sınıf bazlı augmentation ile)
    train_dataset = SportsDataset(
        data_dir=config.TRAIN_DIR,
        transform=train_transform,
        is_train=True,
        class_sample_counts=class_sample_counts,
        mean_sample_count=mean_sample_count
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

