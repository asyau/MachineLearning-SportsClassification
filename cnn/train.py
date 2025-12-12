"""
Ana Training Script - Comprehensive CNN Training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path

from .config import Config
from .dataset import get_data_loaders
from .model import get_model, count_parameters
from .utils import (
    calculate_metrics, top_k_accuracy, save_checkpoint,
    plot_training_history, set_seed, get_device
)
from .augmentation import mixup_data, cutmix_data, mixup_criterion
from sklearn.metrics import accuracy_score
import random


class EarlyStopping:
    """Early stopping callback"""
    
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'min'
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0


def train_epoch(model, train_loader, criterion, optimizer, device, scaler, config):
    """Bir epoch training"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Apply Mixup or CutMix
        use_mixup = config.USE_MIXUP and random.random() < 0.5
        use_cutmix = config.USE_CUTMIX and not use_mixup and random.random() < 0.5
        
        if use_mixup:
            mixed_images, y_a, y_b, lam = mixup_data(images, labels, alpha=config.MIXUP_ALPHA)
        elif use_cutmix:
            mixed_images, y_a, y_b, lam = cutmix_data(images, labels, alpha=config.CUTMIX_ALPHA)
        else:
            mixed_images = images
            y_a = labels
            y_b = None
            lam = 1.0
        
        # Forward pass
        optimizer.zero_grad()
        
        if config.USE_MIXED_PRECISION:
            with autocast('cuda'):
                outputs = model(mixed_images)
                if use_mixup or use_cutmix:
                    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                else:
                    loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            if config.GRADIENT_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(mixed_images)
            if use_mixup or use_cutmix:
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            
            if config.GRADIENT_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            
            optimizer.step()
        
        # Metrics (use original labels for accuracy calculation)
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_a.cpu().numpy())  # Use y_a for metrics
        
        # Progress bar update
        if (batch_idx + 1) % config.PRINT_FREQ == 0:
            pbar.set_postfix({
                'loss': f'{running_loss / (batch_idx + 1):.4f}',
                'acc': f'{accuracy_score(all_labels, all_preds):.4f}'
            })
    
    epoch_loss = running_loss / len(train_loader)
    metrics = calculate_metrics(all_labels, all_preds)
    
    return epoch_loss, metrics


def validate_epoch(model, valid_loader, criterion, device, config):
    """Validation"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(valid_loader, desc='Validation')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            if config.USE_MIXED_PRECISION:
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{running_loss / len(valid_loader):.4f}'
            })
    
    epoch_loss = running_loss / len(valid_loader)
    metrics = calculate_metrics(all_labels, all_preds)
    
    # Top-k accuracy
    top_k_metrics = {}
    for k in config.TOP_K_ACCURACY:
        top_k_metrics[f'top_{k}_acc'] = top_k_accuracy(all_labels, all_probs, k=k)
    
    metrics.update(top_k_metrics)
    
    return epoch_loss, metrics


def get_optimizer(model, config):
    """Optimizer oluştur"""
    if config.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")
    
    return optimizer


def get_scheduler(optimizer, config, num_epochs, num_batches_per_epoch):
    """Learning rate scheduler oluştur"""
    if config.LR_SCHEDULER == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.LR_STEP_SIZE,
            gamma=config.LR_GAMMA
        )
    elif config.LR_SCHEDULER == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=config.LR_MIN
        )
    elif config.LR_SCHEDULER == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=config.LR_MIN
        )
    elif config.LR_SCHEDULER == 'warmup_cosine':
        # Warmup + Cosine annealing
        def lr_lambda(epoch):
            if epoch < config.WARMUP_EPOCHS:
                return (epoch + 1) / config.WARMUP_EPOCHS
            else:
                return 0.5 * (1 + np.cos(np.pi * (epoch - config.WARMUP_EPOCHS) / (num_epochs - config.WARMUP_EPOCHS)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None
    
    return scheduler


def get_criterion(class_weights, device, config):
    """Loss function oluştur"""
    if class_weights is not None and config.USE_CLASS_WEIGHTS:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    
    return criterion


def train(config):
    """Ana training fonksiyonu"""
    print("=" * 80)
    print("SPORTS CLASSIFICATION - CNN TRAINING")
    print("=" * 80)
    
    # Setup
    set_seed(config.SEED, config.DETERMINISTIC)
    device = get_device()
    config.create_dirs()
    
    # Data loaders
    print("\n📊 Loading data...")
    train_loader, valid_loader, test_loader, class_weights = get_data_loaders(config)
    
    # Model
    print(f"\n🤖 Creating model: {config.MODEL_NAME}...")
    model = get_model(
        model_name=config.MODEL_NAME,
        num_classes=config.num_classes,
        pretrained=config.PRETRAINED,
        freeze_backbone=config.FREEZE_BACKBONE
    )
    model = model.to(device)
    
    params = count_parameters(model)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    
    # Loss, Optimizer, Scheduler
    criterion = get_criterion(class_weights, device, config)
    optimizer = get_optimizer(model, config)
    
    num_batches_per_epoch = len(train_loader)
    scheduler = get_scheduler(optimizer, config, config.NUM_EPOCHS, num_batches_per_epoch)
    
    # Mixed precision
    scaler = GradScaler('cuda') if config.USE_MIXED_PRECISION else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        mode='max'
    ) if config.EARLY_STOPPING else None
    
    # TensorBoard
    writer = SummaryWriter(log_dir=config.LOG_DIR) if config.USE_TENSORBOARD else None
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'valid_loss': [],
        'valid_acc': [],
        'valid_f1': [],
        'lr': []
    }
    
    best_valid_acc = 0.0
    start_time = time.time()
    
    print("\n🚀 Starting training...")
    print("=" * 80)
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        print("-" * 80)
        
        # Training
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, config
        )
        
        # Validation
        valid_loss, valid_metrics = validate_epoch(
            model, valid_loader, criterion, device, config
        )
        
        # Learning rate update
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            if config.LR_SCHEDULER == 'plateau':
                scheduler.step(valid_metrics['accuracy'])
            else:
                scheduler.step()
        
        # History update
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1_macro'])
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_metrics['accuracy'])
        history['valid_f1'].append(valid_metrics['f1_macro'])
        history['lr'].append(current_lr)
        
        # Print metrics
        epoch_time = time.time() - epoch_start_time
        print(f"\nTrain - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1_macro']:.4f}")
        print(f"Valid - Loss: {valid_loss:.4f}, Acc: {valid_metrics['accuracy']:.4f}, "
              f"F1: {valid_metrics['f1_macro']:.4f}")
        print(f"LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
        
        # Top-k accuracy
        for k in config.TOP_K_ACCURACY:
            top_k_key = f'top_{k}_acc'
            if top_k_key in valid_metrics:
                print(f"Top-{k} Accuracy: {valid_metrics[top_k_key]:.4f}")
        
        # TensorBoard logging
        if writer:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Valid', valid_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            writer.add_scalar('Accuracy/Valid', valid_metrics['accuracy'], epoch)
            writer.add_scalar('F1/Train', train_metrics['f1_macro'], epoch)
            writer.add_scalar('F1/Valid', valid_metrics['f1_macro'], epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Save checkpoint
        is_best = valid_metrics['accuracy'] > best_valid_acc
        if is_best:
            best_valid_acc = valid_metrics['accuracy']
        
        save_checkpoint(
            model, optimizer, epoch, valid_metrics, config,
            is_best=is_best and config.SAVE_BEST_ONLY,
            is_last=(epoch == config.NUM_EPOCHS) and config.SAVE_LAST
        )
        
        # Early stopping
        if early_stopping:
            early_stopping(valid_metrics['accuracy'])
            if early_stopping.early_stop:
                print(f"\n⚠ Early stopping triggered at epoch {epoch}")
                break
    
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Total time: {total_time / 60:.2f} minutes")
    print(f"Best validation accuracy: {best_valid_acc:.4f}")
    
    # Save training history
    history_path = config.SAVE_DIR / 'training_history.png'
    plot_training_history(history, save_path=history_path)
    
    # Close TensorBoard writer
    if writer:
        writer.close()
    
    return model, history


if __name__ == "__main__":
    from .config import Config
    config = Config()
    model, history = train(config)

