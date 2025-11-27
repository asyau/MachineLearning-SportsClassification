import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def create_train_val_test_split(
    combined_data_dir,
    output_base_dir='data_split',
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_seed=42
):
    """
    Create train/validation/test splits from combined data.
    
    Args:
        combined_data_dir: Directory containing combined sports data
        output_base_dir: Base directory for output (will create train/valid/test inside)
        train_ratio: Ratio for training set (default: 0.8)
        val_ratio: Ratio for validation set (default: 0.1)
        test_ratio: Ratio for test set (default: 0.1)
        random_seed: Random seed for reproducibility (default: 42)
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    base_path = Path(combined_data_dir).parent
    combined_path = Path(combined_data_dir)
    output_path = base_path / output_base_dir
    
    # Create output directories
    train_dir = output_path / 'train'
    valid_dir = output_path / 'valid'
    test_dir = output_path / 'test'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    # Get all sport categories
    sport_categories = [d.name for d in combined_path.iterdir() if d.is_dir()]
    sport_categories.sort()  # Sort for consistency
    
    print("=" * 70)
    print("Creating Train/Validation/Test Splits")
    print("=" * 70)
    print(f"Source: {combined_path}")
    print(f"Output: {output_path}")
    print(f"Split ratios - Train: {train_ratio:.0%}, Val: {val_ratio:.0%}, Test: {test_ratio:.0%}")
    print(f"Random seed: {random_seed}")
    print(f"Number of sport categories: {len(sport_categories)}")
    print("-" * 70)
    
    # Statistics
    total_stats = defaultdict(int)
    split_details = []
    
    # Process each sport category
    for idx, sport in enumerate(sport_categories, 1):
        sport_path = combined_path / sport
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(sport_path.glob(ext)))
        
        if not image_files:
            print(f"[{idx}/{len(sport_categories)}] {sport}: No images found, skipping...")
            continue
        
        # Shuffle images
        random.shuffle(image_files)
        
        total_images = len(image_files)
        
        # Calculate split indices
        train_end = int(total_images * train_ratio)
        val_end = train_end + int(total_images * val_ratio)
        
        # Split the images
        train_images = image_files[:train_end]
        val_images = image_files[train_end:val_end]
        test_images = image_files[val_end:]
        
        # Create sport subdirectories in each split
        train_sport_dir = train_dir / sport
        valid_sport_dir = valid_dir / sport
        test_sport_dir = test_dir / sport
        
        train_sport_dir.mkdir(exist_ok=True)
        valid_sport_dir.mkdir(exist_ok=True)
        test_sport_dir.mkdir(exist_ok=True)
        
        # Copy images to respective directories with new naming
        for idx, img in enumerate(train_images, 1):
            new_name = f"train_{idx:04d}{img.suffix}"
            shutil.copy2(img, train_sport_dir / new_name)
        
        for idx, img in enumerate(val_images, 1):
            new_name = f"valid_{idx:04d}{img.suffix}"
            shutil.copy2(img, valid_sport_dir / new_name)
        
        for idx, img in enumerate(test_images, 1):
            new_name = f"test_{idx:04d}{img.suffix}"
            shutil.copy2(img, test_sport_dir / new_name)
        
        # Update statistics
        total_stats['train'] += len(train_images)
        total_stats['valid'] += len(val_images)
        total_stats['test'] += len(test_images)
        total_stats['total'] += total_images
        
        split_details.append({
            'sport': sport,
            'total': total_images,
            'train': len(train_images),
            'valid': len(val_images),
            'test': len(test_images)
        })
        
        print(f"[{idx:3d}/{len(sport_categories)}] {sport:30s} | "
              f"Total: {total_images:3d} → "
              f"Train: {len(train_images):3d}, "
              f"Val: {len(val_images):3d}, "
              f"Test: {len(test_images):3d}")
    
    print("-" * 70)
    print("\n" + "=" * 70)
    print("Split Summary")
    print("=" * 70)
    print(f"Total images processed: {total_stats['total']:,}")
    print(f"  - Training set:   {total_stats['train']:,} images "
          f"({total_stats['train']/total_stats['total']*100:.1f}%)")
    print(f"  - Validation set: {total_stats['valid']:,} images "
          f"({total_stats['valid']/total_stats['total']*100:.1f}%)")
    print(f"  - Test set:       {total_stats['test']:,} images "
          f"({total_stats['test']/total_stats['total']*100:.1f}%)")
    print("=" * 70)
    
    # Save split information to a text file
    info_file = output_path / 'split_info.txt'
    with open(info_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Train/Validation/Test Split Information\n")
        f.write("=" * 70 + "\n")
        f.write(f"Random seed: {random_seed}\n")
        f.write(f"Split ratios - Train: {train_ratio:.0%}, Val: {val_ratio:.0%}, Test: {test_ratio:.0%}\n")
        f.write(f"Total categories: {len(sport_categories)}\n")
        f.write(f"Total images: {total_stats['total']:,}\n")
        f.write("-" * 70 + "\n\n")
        
        f.write("Per-Category Breakdown:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Sport':<30} {'Total':>7} {'Train':>7} {'Valid':>7} {'Test':>7}\n")
        f.write("-" * 70 + "\n")
        
        for detail in split_details:
            f.write(f"{detail['sport']:<30} "
                   f"{detail['total']:>7} "
                   f"{detail['train']:>7} "
                   f"{detail['valid']:>7} "
                   f"{detail['test']:>7}\n")
        
        f.write("-" * 70 + "\n")
        f.write(f"{'TOTAL':<30} "
               f"{total_stats['total']:>7} "
               f"{total_stats['train']:>7} "
               f"{total_stats['valid']:>7} "
               f"{total_stats['test']:>7}\n")
        f.write("=" * 70 + "\n")
    
    print(f"\n✓ Split complete!")
    print(f"✓ Data organized in: {output_path}")
    print(f"✓ Split details saved to: {info_file}")
    print("\nFolder structure:")
    print(f"  {output_path}/")
    print(f"    ├── train/")
    print(f"    │   ├── {sport_categories[0]}/")
    print(f"    │   ├── {sport_categories[1]}/")
    print(f"    │   └── ... ({len(sport_categories)} categories)")
    print(f"    ├── valid/")
    print(f"    │   └── ... ({len(sport_categories)} categories)")
    print(f"    └── test/")
    print(f"        └── ... ({len(sport_categories)} categories)")

if __name__ == "__main__":
    # Set the base directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    combined_data_directory = os.path.join(script_dir, 'combined_data')
    
    # Create the splits
    create_train_val_test_split(
        combined_data_dir=combined_data_directory,
        output_base_dir='data_split',
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42
    )
    
    print("\n" + "=" * 70)
    print("Ready for model training!")
    print("=" * 70)

