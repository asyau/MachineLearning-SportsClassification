"""
Get Training Set Class Sizes
Counts the number of images in each class in the training set
and exports to a text file.
"""

import os
from pathlib import Path

def count_train_class_sizes(train_dir='train', output_file='train_class_sizes.txt'):
    """
    Count images in each class of the training set and save to file.
    
    Args:
        train_dir: Path to the training directory
        output_file: Output text file name
    """
    train_path = Path(train_dir)
    
    if not train_path.exists():
        print(f"Error: Training directory '{train_dir}' not found!")
        return
    
    # Dictionary to store class sizes
    class_sizes = {}
    
    # Valid image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.JPG', '.JPEG', '.PNG'}
    
    # Count images in each class directory
    for class_dir in sorted(train_path.iterdir()):
        if class_dir.is_dir():
            class_name = class_dir.name
            
            # Count image files
            image_count = sum(1 for file in class_dir.iterdir() 
                            if file.suffix in image_extensions)
            
            class_sizes[class_name] = image_count
    
    # Sort by class name
    sorted_classes = sorted(class_sizes.items())
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Training Set Class Sizes\n")
        f.write("=" * 60 + "\n\n")
        
        for class_name, count in sorted_classes:
            f.write(f"{class_name} - {count}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Total Classes: {len(class_sizes)}\n")
        f.write(f"Total Images: {sum(class_sizes.values())}\n")
        f.write("=" * 60 + "\n")
    
    # Print summary
    print("=" * 60)
    print("Training Set Class Sizes")
    print("=" * 60)
    print(f"✓ Found {len(class_sizes)} classes")
    print(f"✓ Total images: {sum(class_sizes.values()):,}")
    print(f"✓ Results saved to: {output_file}")
    print("=" * 60)
    
    # Show first 10 and last 10 classes
    print("\nFirst 10 classes:")
    for class_name, count in sorted_classes[:10]:
        print(f"  {class_name} - {count}")
    
    print("\nLast 10 classes:")
    for class_name, count in sorted_classes[-10:]:
        print(f"  {class_name} - {count}")
    
    return class_sizes

if __name__ == "__main__":
    # Get script directory
    script_dir = Path(__file__).parent
    train_dir = script_dir / 'train'
    output_file = script_dir / 'train_class_sizes.txt'
    
    # Count and export class sizes
    class_sizes = count_train_class_sizes(train_dir, output_file)

