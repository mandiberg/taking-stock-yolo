import shutil
from pathlib import Path
import random

def split_dataset(source_images, source_labels, output_dir, train_ratio=0.8):
    """Split dataset into train/val"""
    
    # Get all images
    images = list(Path(source_images).glob('*.jpg')) + \
             list(Path(source_images).glob('*.png')) + \
             list(Path(source_images).glob('*.jpeg'))
    
    # Shuffle
    random.seed(42)  # For reproducibility
    random.shuffle(images)
    
    # Split
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    print(f"Total images: {len(images)}")
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")
    
    # Create directories
    for split in ['train', 'val']:
        (Path(output_dir) / 'images' / split).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Copy train files
    for img in train_images:
        shutil.copy(img, Path(output_dir) / 'images' / 'train' / img.name)
        label = Path(source_labels) / f"{img.stem}.txt"
        if label.exists():
            shutil.copy(label, Path(output_dir) / 'labels' / 'train' / label.name)
    
    # Copy val files
    for img in val_images:
        shutil.copy(img, Path(output_dir) / 'images' / 'val' / img.name)
        label = Path(source_labels) / f"{img.stem}.txt"
        if label.exists():
            shutil.copy(label, Path(output_dir) / 'labels' / 'val' / label.name)

# Usage - adjust paths to your export
split_dataset(
    source_images='/Users/michaelmandiberg/Downloads/project-2-at-2025-10-29-16-40-56110721/images',
    source_labels='/Users/michaelmandiberg/Downloads/project-2-at-2025-10-29-16-40-56110721/labels',
    output_dir='yolo_dataset',
    train_ratio=0.8
)