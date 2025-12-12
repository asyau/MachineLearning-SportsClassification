"""
CNN Model tanımları - Transfer Learning desteği ile
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ResNet50_Weights, ResNet101_Weights,
    EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights,
    EfficientNet_B3_Weights, EfficientNet_B4_Weights,
    ViT_B_16_Weights
)
from .config import Config


def get_model(model_name='efficientnet_b3', num_classes=100, pretrained=True, freeze_backbone=False):
    """
    Model oluştur
    
    Args:
        model_name: Model adı ('resnet50', 'efficientnet_b0-b7', 'vit_b_16')
        num_classes: Sınıf sayısı
        pretrained: Pretrained weights kullan
        freeze_backbone: Backbone'u dondur (sadece classifier eğit)
    
    Returns:
        torch.nn.Module: Model
    """
    model = None
    
    # ResNet models
    if model_name == 'resnet50':
        if pretrained:
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            model = models.resnet50(weights=None)
        
        # Classifier'ı değiştir
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
    
    elif model_name == 'resnet101':
        if pretrained:
            model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        else:
            model = models.resnet101(weights=None)
        
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
    
    # EfficientNet models
    elif model_name == 'efficientnet_b0':
        if pretrained:
            model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_b0(weights=None)
        
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
    
    elif model_name == 'efficientnet_b1':
        if pretrained:
            model = models.efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V2)
        else:
            model = models.efficientnet_b1(weights=None)
        
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
    
    elif model_name == 'efficientnet_b2':
        if pretrained:
            model = models.efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_b2(weights=None)
        
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
    
    elif model_name == 'efficientnet_b3':
        if pretrained:
            model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_b3(weights=None)
        
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
    
    elif model_name == 'efficientnet_b4':
        if pretrained:
            model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_b4(weights=None)
        
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
    
    # Vision Transformer
    elif model_name == 'vit_b_16':
        if pretrained:
            model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            model = models.vit_b_16(weights=None)
        
        num_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.heads.parameters():
                param.requires_grad = True
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model


def count_parameters(model):
    """Model parametre sayısını hesapla"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }


if __name__ == "__main__":
    # Test
    from .config import Config
    config = Config()
    model = get_model(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED,
        freeze_backbone=config.FREEZE_BACKBONE
    )
    
    params = count_parameters(model)
    print(f"\nModel: {config.MODEL_NAME}")
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Frozen parameters: {params['frozen']:,}")
    
    # Test forward pass
    x = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    with torch.no_grad():
        output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

