"""
Data Augmentation for Small Classes
Augments training classes with less than 100 images to reach 110 images.
Uses various augmentation techniques: rotation, flipping, brightness, contrast, etc.
"""

import os
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import shutil

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class ImageAugmentor:
    """Class to handle various image augmentation techniques."""
    
    def __init__(self):
        self.augmentation_methods = [
            self.rotate,
            self.flip_horizontal,
            self.flip_vertical,
            self.adjust_brightness,
            self.adjust_contrast,
            self.adjust_saturation,
            self.add_noise,
            self.blur,
            self.sharpen,
            self.zoom,
        ]
    
    def rotate(self, img):
        """Rotate image by a random angle between -30 and 30 degrees."""
        angle = random.uniform(-30, 30)
        return img.rotate(angle, fillcolor='white', expand=False)
    
    def flip_horizontal(self, img):
        """Flip image horizontally."""
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    
    def flip_vertical(self, img):
        """Flip image vertically."""
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    
    def adjust_brightness(self, img):
        """Adjust image brightness."""
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)
    
    def adjust_contrast(self, img):
        """Adjust image contrast."""
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    
    def adjust_saturation(self, img):
        """Adjust image saturation."""
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)
    
    def add_noise(self, img):
        """Add random noise to the image."""
        img_array = np.array(img)
        noise = np.random.normal(0, 10, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    
    def blur(self, img):
        """Apply Gaussian blur."""
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
    
    def sharpen(self, img):
        """Sharpen the image."""
        return img.filter(ImageFilter.SHARPEN)
    
    def zoom(self, img):
        """Zoom in/out on the image."""
        width, height = img.size
        zoom_factor = random.uniform(0.8, 1.2)
        
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)
        
        # Resize
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Crop or pad to original size
        if zoom_factor > 1:
            # Crop center
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            return img_resized.crop((left, top, left + width, top + height))
        else:
            # Pad with white
            new_img = Image.new('RGB', (width, height), 'white')
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            new_img.paste(img_resized, (left, top))
            return new_img
    
    def augment(self, img, num_augmentations=2):
        """
        Apply random augmentations to an image.
        
        Args:
            img: PIL Image
            num_augmentations: Number of augmentation techniques to apply
            
        Returns:
            Augmented PIL Image
        """
        # Randomly select augmentation methods
        methods = random.sample(self.augmentation_methods, 
                              min(num_augmentations, len(self.augmentation_methods)))
        
        augmented_img = img.copy()
        for method in methods:
            try:
                augmented_img = method(augmented_img)
            except Exception as e:
                print(f"Warning: Augmentation failed: {e}")
                continue
        
        return augmented_img


def get_image_files(directory):
    """Get all image files in a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.JPG', '.JPEG', '.PNG'}
    return [f for f in Path(directory).iterdir() 
            if f.is_file() and f.suffix in image_extensions]


def augment_class(class_dir, target_count=110, augmentor=None):
    """
    Augment images in a class directory to reach target count.
    
    Args:
        class_dir: Path to class directory
        target_count: Target number of images
        augmentor: ImageAugmentor instance
        
    Returns:
        Number of augmented images created
    """
    if augmentor is None:
        augmentor = ImageAugmentor()
    
    class_path = Path(class_dir)
    
    # Get original images (excluding already augmented ones)
    all_images = get_image_files(class_path)
    original_images = [img for img in all_images if not img.stem.startswith('aug_')]
    
    current_count = len(all_images)
    needed = target_count - current_count
    
    if needed <= 0:
        return 0
    
    print(f"  Augmenting {class_path.name}: {current_count} → {target_count} (+{needed} images)")
    
    # Create augmented images
    augmented_count = 0
    attempts = 0
    max_attempts = needed * 3  # Prevent infinite loops
    
    while augmented_count < needed and attempts < max_attempts:
        attempts += 1
        
        # Randomly select an original image
        source_img_path = random.choice(original_images)
        
        try:
            # Load image
            img = Image.open(source_img_path)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Apply augmentation
            num_augmentations = random.randint(2, 4)
            augmented_img = augmentor.augment(img, num_augmentations)
            
            # Generate unique filename
            aug_filename = f"aug_{augmented_count + 1:04d}{source_img_path.suffix}"
            aug_path = class_path / aug_filename
            
            # Save augmented image
            augmented_img.save(aug_path, quality=95)
            augmented_count += 1
            
        except Exception as e:
            print(f"    Warning: Failed to augment {source_img_path.name}: {e}")
            continue
    
    if augmented_count < needed:
        print(f"    Warning: Could only create {augmented_count}/{needed} augmented images")
    
    return augmented_count


def augment_small_classes(train_dir='train', threshold=100, target=110):
    """
    Augment all classes with less than threshold images to reach target count.
    
    Args:
        train_dir: Path to training directory
        threshold: Classes with fewer images than this will be augmented
        target: Target number of images per class
        
    Returns:
        Dictionary with augmentation statistics
    """
    train_path = Path(train_dir)
    
    if not train_path.exists():
        print(f"Error: Training directory '{train_dir}' not found!")
        return {}
    
    print("=" * 70)
    print("Data Augmentation for Small Classes")
    print("=" * 70)
    print(f"Threshold: {threshold} images")
    print(f"Target: {target} images")
    print(f"Training directory: {train_path}")
    print("-" * 70)
    
    # Identify classes that need augmentation
    augmentor = ImageAugmentor()
    classes_to_augment = []
    
    for class_dir in sorted(train_path.iterdir()):
        if not class_dir.is_dir():
            continue
        
        image_files = get_image_files(class_dir)
        count = len(image_files)
        
        if count < threshold:
            classes_to_augment.append((class_dir, count))
    
    if not classes_to_augment:
        print("✓ No classes need augmentation!")
        return {}
    
    print(f"\nFound {len(classes_to_augment)} classes needing augmentation:\n")
    
    # Show classes that will be augmented
    for class_dir, count in classes_to_augment:
        print(f"  {class_dir.name:<30s} : {count:3d} images → {target} images")
    
    print("\n" + "-" * 70)
    
    # Confirm before proceeding
    response = input(f"\nAugment {len(classes_to_augment)} classes? (y/n): ").strip().lower()
    if response != 'y':
        print("Augmentation cancelled.")
        return {}
    
    print("\n" + "=" * 70)
    print("Starting Augmentation...")
    print("=" * 70 + "\n")
    
    # Augment each class
    stats = {}
    total_augmented = 0
    
    for idx, (class_dir, original_count) in enumerate(classes_to_augment, 1):
        print(f"[{idx}/{len(classes_to_augment)}]", end=" ")
        
        augmented = augment_class(class_dir, target, augmentor)
        
        stats[class_dir.name] = {
            'original': original_count,
            'augmented': augmented,
            'total': original_count + augmented
        }
        
        total_augmented += augmented
    
    # Summary
    print("\n" + "=" * 70)
    print("Augmentation Complete!")
    print("=" * 70)
    print(f"Classes augmented: {len(classes_to_augment)}")
    print(f"Total augmented images created: {total_augmented}")
    print("=" * 70)
    
    # Save report
    report_path = train_path.parent / 'augmentation_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Data Augmentation Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Threshold: {threshold} images\n")
        f.write(f"Target: {target} images\n")
        f.write(f"Classes augmented: {len(classes_to_augment)}\n")
        f.write(f"Total augmented images: {total_augmented}\n\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Class Name':<30s} {'Original':>10s} {'Augmented':>12s} {'Total':>10s}\n")
        f.write("-" * 70 + "\n")
        
        for class_name, stat in sorted(stats.items()):
            f.write(f"{class_name:<30s} {stat['original']:>10d} "
                   f"{stat['augmented']:>12d} {stat['total']:>10d}\n")
        
        f.write("=" * 70 + "\n")
    
    print(f"\n✓ Report saved to: {report_path}")
    
    return stats


if __name__ == "__main__":
    # Configuration
    TRAIN_DIR = Path(__file__).parent / 'train'
    THRESHOLD = 100  # Augment classes with fewer than this many images
    TARGET = 110     # Target number of images per class
    
    # Run augmentation
    stats = augment_small_classes(
        train_dir=TRAIN_DIR,
        threshold=THRESHOLD,
        target=TARGET
    )
    
    print("\n" + "=" * 70)
    print("Ready for training!")
    print("=" * 70)

