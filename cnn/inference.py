"""
Inference Script - Tek görüntü veya batch inference
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
import argparse

from .config import Config
from .dataset import get_transforms, SportsDataset
from .model import get_model
from .utils import load_checkpoint, set_seed, get_device


def predict_image(model, image_path, device, config, class_names, top_k=5):
    """
    Tek bir görüntü için tahmin yap
    
    Args:
        model: Model
        image_path: Görüntü dosya yolu
        device: Device
        config: Config
        class_names: Sınıf isimleri dict
        top_k: Top-k tahmin göster
    
    Returns:
        dict: Tahminler ve olasılıklar
    """
    # Görüntüyü yükle ve transform et
    transform = get_transforms(
        image_size=config.IMAGE_SIZE,
        is_train=False
    )
    
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {e}")
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Tahmin
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        probs, indices = torch.topk(probs, top_k, dim=1)
    
    # Sonuçları formatla
    probs = probs.cpu().numpy()[0]
    indices = indices.cpu().numpy()[0]
    
    predictions = []
    for i, (prob, idx) in enumerate(zip(probs, indices)):
        predictions.append({
            'rank': i + 1,
            'class': class_names[idx],
            'probability': float(prob)
        })
    
    return {
        'image_path': str(image_path),
        'predictions': predictions
    }


def predict_batch(model, image_paths, device, config, class_names, top_k=5):
    """
    Birden fazla görüntü için batch tahmin
    
    Args:
        model: Model
        image_paths: Görüntü dosya yolları listesi
        device: Device
        config: Config
        class_names: Sınıf isimleri dict
        top_k: Top-k tahmin göster
    
    Returns:
        list: Her görüntü için tahminler
    """
    transform = get_transforms(
        image_size=config.IMAGE_SIZE,
        is_train=False
    )
    
    # Görüntüleri yükle
    images = []
    valid_paths = []
    
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image)
            images.append(image_tensor)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Warning: Skipping {img_path}: {e}")
    
    if not images:
        raise ValueError("No valid images found!")
    
    # Batch oluştur
    batch = torch.stack(images).to(device)
    
    # Tahmin
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
        probs = F.softmax(outputs, dim=1)
        probs, indices = torch.topk(probs, top_k, dim=1)
    
    # Sonuçları formatla
    probs = probs.cpu().numpy()
    indices = indices.cpu().numpy()
    
    results = []
    for img_path, img_probs, img_indices in zip(valid_paths, probs, indices):
        predictions = []
        for i, (prob, idx) in enumerate(zip(img_probs, img_indices)):
            predictions.append({
                'rank': i + 1,
                'class': class_names[idx],
                'probability': float(prob)
            })
        
        results.append({
            'image_path': str(img_path),
            'predictions': predictions
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Sports Classification Inference')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--dir', type=str, help='Directory containing images')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path (default: best_model.pt)')
    parser.add_argument('--top_k', type=int, default=5, help='Top-k predictions to show')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for directory inference')
    
    args = parser.parse_args()
    
    # Config
    config = Config()
    set_seed(config.SEED, config.DETERMINISTIC)
    device = get_device()
    config.create_dirs()
    
    # Class names yükle (dataset'ten)
    train_dataset = SportsDataset(config.TRAIN_DIR, transform=None, is_train=False)
    class_names = train_dataset.idx_to_class
    config.num_classes = len(class_names)
    
    # Model
    print(f"Loading model: {config.MODEL_NAME}...")
    model = get_model(
        model_name=config.MODEL_NAME,
        num_classes=config.num_classes,
        pretrained=False
    )
    model = model.to(device)
    
    # Checkpoint yükle
    checkpoint_path = args.checkpoint or config.get_best_model_path()
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(model, None, checkpoint_path, device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Inference
    if args.image:
        # Single image
        print(f"\nPredicting: {args.image}")
        result = predict_image(model, args.image, device, config, class_names, top_k=args.top_k)
        
        print("\n" + "=" * 80)
        print("PREDICTION RESULTS")
        print("=" * 80)
        print(f"Image: {result['image_path']}")
        print("\nTop Predictions:")
        for pred in result['predictions']:
            print(f"  {pred['rank']}. {pred['class']:30s} - {pred['probability']*100:.2f}%")
        print("=" * 80)
    
    elif args.dir:
        # Directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        image_paths = [
            p for p in Path(args.dir).rglob('*')
            if p.suffix in image_extensions
        ]
        
        if not image_paths:
            print(f"No images found in {args.dir}")
            return
        
        print(f"\nFound {len(image_paths)} images")
        
        # Batch inference
        batch_size = args.batch_size or config.INFERENCE_BATCH_SIZE
        all_results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1}...")
            results = predict_batch(model, batch_paths, device, config, class_names, top_k=args.top_k)
            all_results.extend(results)
        
        # Sonuçları yazdır
        print("\n" + "=" * 80)
        print("BATCH PREDICTION RESULTS")
        print("=" * 80)
        for result in all_results:
            print(f"\n{result['image_path']}:")
            for pred in result['predictions']:
                print(f"  {pred['rank']}. {pred['class']:30s} - {pred['probability']*100:.2f}%")
        print("=" * 80)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

