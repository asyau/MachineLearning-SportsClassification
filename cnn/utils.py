"""
CNN için yardımcı fonksiyonlar: metrics, visualization, checkpointing, logging
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import defaultdict
from .config import Config


def calculate_metrics(y_true, y_pred, num_classes=None):
    """
    Classification metriklerini hesapla
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        num_classes: Sınıf sayısı
    
    Returns:
        dict: Metrikler
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1 (macro ve weighted)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'support': support.tolist(),
    }
    
    return metrics


def top_k_accuracy(y_true, y_pred_proba, k=5):
    """
    Top-k accuracy hesapla
    
    Args:
        y_true: Gerçek etiketler
        y_pred_proba: Olasılık tahminleri (N x num_classes)
        k: Top-k değeri
    
    Returns:
        float: Top-k accuracy
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Top-k indeksleri
    top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
    
    # Her örnek için doğru sınıf top-k içinde mi?
    correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
    
    return correct.mean()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, figsize=(20, 20)):
    """
    Confusion matrix çiz
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        class_names: Sınıf isimleri (dict veya list)
        save_path: Kayıt yolu
        figsize: Figür boyutu
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize et
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
    
    # Absolute values
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[0], cbar_kws={'label': 'Count'}
    )
    axes[0].set_title('Confusion Matrix (Absolute)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('True', fontsize=12)
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')
    plt.setp(axes[0].get_yticklabels(), rotation=0)
    
    # Normalized
    sns.heatmap(
        cm_normalized, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[1], cbar_kws={'label': 'Normalized'}
    )
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('True', fontsize=12)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
    plt.setp(axes[1].get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.close()


def plot_training_history(history, save_path=None):
    """
    Training history grafiklerini çiz
    
    Args:
        history: Training history dict
        save_path: Kayıt yolu
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['valid_loss'], 'r-', label='Valid Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, history['valid_acc'], 'r-', label='Valid Accuracy', linewidth=2)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    if 'lr' in history:
        axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # F1 Score
    if 'train_f1' in history:
        axes[1, 1].plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
        axes[1, 1].plot(epochs, history['valid_f1'], 'r-', label='Valid F1', linewidth=2)
        axes[1, 1].set_title('Training and Validation F1 Score', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to: {save_path}")
    
    plt.close()


def save_checkpoint(model, optimizer, epoch, metrics, config, is_best=False, is_last=False):
    """
    Model checkpoint kaydet
    
    Args:
        model: Model
        optimizer: Optimizer
        epoch: Epoch numarası
        metrics: Metrikler dict
        config: Config
        is_best: En iyi model mi?
        is_last: Son model mi?
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config.get_config_dict(),
    }
    
    if is_best:
        torch.save(checkpoint, config.get_best_model_path())
        print(f"✓ Best model saved (epoch {epoch}, acc: {metrics['accuracy']:.4f})")
    
    if is_last:
        torch.save(checkpoint, config.get_last_model_path())
        print(f"✓ Last model saved (epoch {epoch})")


def load_checkpoint(model, optimizer, checkpoint_path, device='cuda'):
    """
    Model checkpoint yükle
    
    Args:
        model: Model
        optimizer: Optimizer
        checkpoint_path: Checkpoint dosya yolu
        device: Device
    
    Returns:
        dict: Checkpoint bilgileri
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def save_metrics_report(metrics, class_names, save_path):
    """
    Detaylı metrik raporu kaydet
    
    Args:
        metrics: Metrikler dict
        class_names: Sınıf isimleri
        save_path: Kayıt yolu
    """
    report = classification_report(
        metrics.get('y_true', []),
        metrics.get('y_pred', []),
        target_names=[class_names[i] for i in range(len(class_names))],
        output_dict=True
    )
    
    # JSON olarak kaydet
    with open(save_path, 'w') as f:
        json.dump({
            'overall': {
                'accuracy': metrics['accuracy'],
                'precision_macro': metrics['precision_macro'],
                'recall_macro': metrics['recall_macro'],
                'f1_macro': metrics['f1_macro'],
                'precision_weighted': metrics['precision_weighted'],
                'recall_weighted': metrics['recall_weighted'],
                'f1_weighted': metrics['f1_weighted'],
            },
            'per_class': report
        }, f, indent=2)
    
    print(f"Metrics report saved to: {save_path}")


def set_seed(seed=42, deterministic=True):
    """Reproducibility için seed ayarla"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device():
    """CUDA kullanılabilir mi kontrol et"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

