"""
Advanced Data Augmentation: Mixup and CutMix
"""

import torch
import numpy as np


def mixup_data(x, y, alpha=1.0):
    """
    Mixup augmentation
    
    Args:
        x: Input images (batch_size, channels, height, width)
        y: Labels (batch_size,)
        alpha: Mixup alpha parameter
    
    Returns:
        mixed_x: Mixed images
        y_a, y_b: Original labels
        lam: Mixup lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """
    CutMix augmentation
    
    Args:
        x: Input images (batch_size, channels, height, width)
        y: Labels (batch_size,)
        alpha: CutMix alpha parameter
    
    Returns:
        mixed_x: Mixed images
        y_a, y_b: Original labels
        lam: CutMix lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # Get random bounding box
    W, H = x.size(3), x.size(2)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Random center
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # Calculate bounding box
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply CutMix
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup loss calculation
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a, y_b: Mixed labels
        lam: Mixup lambda
    
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


