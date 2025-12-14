import shutil
from pathlib import Path
import random
import os
import yaml

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
SORTED_IMAGES_FOLDER = "/Volumes/OWC52/YOLO_Training_Data/sorted_images"
YOLO_READY_DATASET_FOLDER = "/Users/michael.mandiberg/Documents/GitHub/taking-stock-yolo/yolo_dataset"

# get all folders in SORTED_IMAGES_FOLDER
folders = [f.path for f in os.scandir(SORTED_IMAGES_FOLDER) if f.is_dir()]

# Collect class information
class_names = {}

for folder in folders:
    class_id, class_name = os.path.basename(folder).split('_', 1)
    this_folder = os.path.join(SORTED_IMAGES_FOLDER,folder)
    print(f"Processing folder: {this_folder}")
    this_folder_images = os.path.join(this_folder, 'images')
    
    # Check if images folder exists and has files
    if not os.path.exists(this_folder_images):
        print(f"Images folder doesn't exist: {this_folder_images}, skipping.")
        continue
    
    files = os.listdir(this_folder_images)
    if len(files) == 0:
        print(f"No images found in {this_folder_images}, skipping.")
        continue
    if class_id.isdigit():
        # Store class information
        class_names[int(class_id)] = class_name

    split_dataset(
        source_images= this_folder_images,
        source_labels=os.path.join(this_folder, 'labels'),
        output_dir=YOLO_READY_DATASET_FOLDER,
        train_ratio=0.8
    )

# Create YOLO data.yaml file
yaml_data = {
    'path': YOLO_READY_DATASET_FOLDER,
    'train': 'images/train',
    'val': 'images/val',
    'nc': len(class_names),
    'names': class_names
}

yaml_path = os.path.join(YOLO_READY_DATASET_FOLDER, 'data.yaml')
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=True)

print(f"\nCreated YOLO config file: {yaml_path}")
print(f"Total classes: {len(class_names)}")
print(f"Classes: {class_names}")