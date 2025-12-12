"""
Model Evaluation Script - Comprehensive evaluation with metrics and visualizations
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path

from .config import Config
from .dataset import get_data_loaders
from .model import get_model
from .utils import (
    calculate_metrics, top_k_accuracy, plot_confusion_matrix,
    save_metrics_report, load_checkpoint, set_seed, get_device
)


def evaluate_model(model, data_loader, device, config, class_names):
    """
    Model'i değerlendir
    
    Args:
        model: Model
        data_loader: Data loader
        device: Device
        config: Config
        class_names: Sınıf isimleri
    
    Returns:
        dict: Metrikler ve tahminler
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Evaluating')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Metrikleri hesapla
    metrics = calculate_metrics(all_labels, all_preds)
    
    # Top-k accuracy
    top_k_metrics = {}
    for k in config.TOP_K_ACCURACY:
        top_k_metrics[f'top_{k}_acc'] = top_k_accuracy(all_labels, all_probs, k=k)
    
    metrics.update(top_k_metrics)
    
    # Tahminleri ekle
    metrics['y_true'] = all_labels
    metrics['y_pred'] = all_preds
    metrics['y_proba'] = all_probs
    
    return metrics


def print_metrics(metrics, split_name='Test'):
    """Metrikleri yazdır"""
    print("\n" + "=" * 80)
    print(f"{split_name.upper()} SET METRICS")
    print("=" * 80)
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Precision (Macro):  {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro):     {metrics['recall_macro']:.4f}")
    print(f"F1 Score (Macro):   {metrics['f1_macro']:.4f}")
    print(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (Weighted):    {metrics['recall_weighted']:.4f}")
    print(f"F1 Score (Weighted):  {metrics['f1_weighted']:.4f}")
    
    # Top-k accuracy
    for k in [1, 3, 5]:
        top_k_key = f'top_{k}_acc'
        if top_k_key in metrics:
            print(f"Top-{k} Accuracy:     {metrics[top_k_key]:.4f}")
    
    print("=" * 80)


def evaluate(config, checkpoint_path=None, split='test'):
    """
    Model'i değerlendir
    
    Args:
        config: Config
        checkpoint_path: Checkpoint dosya yolu (None ise best model)
        split: 'test' veya 'valid'
    """
    print("=" * 80)
    print("SPORTS CLASSIFICATION - MODEL EVALUATION")
    print("=" * 80)
    
    # Setup
    set_seed(config.SEED, config.DETERMINISTIC)
    device = get_device()
    config.create_dirs()
    
    # Data loader
    print(f"\n📊 Loading {split} data...")
    train_loader, valid_loader, test_loader, _ = get_data_loaders(config)
    
    if split == 'test':
        data_loader = test_loader
    elif split == 'valid':
        data_loader = valid_loader
    else:
        raise ValueError(f"Unknown split: {split}")
    
    # Model
    print(f"\n🤖 Loading model: {config.MODEL_NAME}...")
    model = get_model(
        model_name=config.MODEL_NAME,
        num_classes=config.num_classes,
        pretrained=False  # Checkpoint'ten yüklenecek
    )
    model = model.to(device)
    
    # Checkpoint yükle
    if checkpoint_path is None:
        checkpoint_path = config.get_best_model_path()
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(model, None, checkpoint_path, device)
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Checkpoint metrics: {checkpoint['metrics']}")
    
    # Class names
    class_names = config.class_names
    class_names_list = [class_names[i] for i in range(len(class_names))]
    
    # Evaluate
    print(f"\n🔍 Evaluating on {split} set...")
    metrics = evaluate_model(model, data_loader, device, config, class_names_list)
    
    # Print metrics
    print_metrics(metrics, split_name=split)
    
    # Confusion matrix
    print("\n📊 Generating confusion matrix...")
    cm_path = config.SAVE_DIR / f'confusion_matrix_{split}.png'
    plot_confusion_matrix(
        metrics['y_true'],
        metrics['y_pred'],
        class_names_list,
        save_path=cm_path,
        figsize=(25, 25)
    )
    
    # Detailed metrics report
    print("\n📝 Saving detailed metrics report...")
    report_path = config.SAVE_DIR / f'metrics_report_{split}.json'
    save_metrics_report(metrics, class_names_list, report_path)
    
    # Per-class performance (en iyi ve en kötü)
    print("\n📈 Class Performance Analysis:")
    print("-" * 80)
    
    f1_per_class = np.array(metrics['f1_per_class'])
    support = np.array(metrics['support'])
    
    # En iyi 10 sınıf
    top_10_indices = np.argsort(f1_per_class)[-10:][::-1]
    print("\nTop 10 Classes (by F1 Score):")
    for idx in top_10_indices:
        class_name = class_names_list[idx]
        print(f"  {class_name:30s} - F1: {f1_per_class[idx]:.4f}, Support: {support[idx]}")
    
    # En kötü 10 sınıf
    bottom_10_indices = np.argsort(f1_per_class)[:10]
    print("\nBottom 10 Classes (by F1 Score):")
    for idx in bottom_10_indices:
        class_name = class_names_list[idx]
        print(f"  {class_name:30s} - F1: {f1_per_class[idx]:.4f}, Support: {support[idx]}")
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    
    return metrics


if __name__ == "__main__":
    from .config import Config
    config = Config()
    
    # Test set evaluation
    test_metrics = evaluate(config, split='test')
    
    # Validation set evaluation (opsiyonel)
    # valid_metrics = evaluate(config, split='valid')

