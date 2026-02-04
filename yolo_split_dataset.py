import shutil
from pathlib import Path
import random
import os
import yaml

def split_dataset(source_images, source_labels, output_dir, class_id_to_YOLOid=None, train_ratio=0.8):
    """Split dataset into train/val"""
    
    def convert_label_file(input_path, output_path, class_id_to_YOLOid):
        """Read label file, convert class IDs to YOLO IDs, and write to output"""
        with open(input_path, 'r') as f:
            lines = f.readlines()
        
        converted_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                try:
                    old_class_id = parts[0]
                    # Convert to YOLO ID
                    yolo_id = class_id_to_YOLOid.get(old_class_id, old_class_id)
                    parts[0] = str(yolo_id)
                    parts[3] = '0.99' if float(parts[3]) >= 1 else parts[3]  # width
                    parts[4] = '0.99' if float(parts[4]) >= 1 else parts[4]  # height
                    converted_lines.append(' '.join(parts) + '\n')
                except Exception as e:
                    print(f"Error converting line '{line.strip()}' in {input_path}: {e}")
        with open(output_path, 'w') as f:
            f.writelines(converted_lines)
    
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
            output_label = Path(output_dir) / 'labels' / 'train' / label.name
            convert_label_file(label, output_label, class_id_to_YOLOid)
    
    # Copy val files
    for img in val_images:
        shutil.copy(img, Path(output_dir) / 'images' / 'val' / img.name)
        label = Path(source_labels) / f"{img.stem}.txt"
        if label.exists():
            output_label = Path(output_dir) / 'labels' / 'val' / label.name
            convert_label_file(label, output_label, class_id_to_YOLOid)

# Usage - adjust paths to your export
SORTED_IMAGES_FOLDER = os.path.join(os.path.expanduser("~"), "Documents/YOLO_Training_Data/sorted_images")
YOLO_READY_DATASET_FOLDER = os.path.join(os.path.expanduser("~"), "Documents/GitHub/taking-stock-yolo/yolo_dataset")

# get all folders in SORTED_IMAGES_FOLDER
folders = [f.path for f in os.scandir(SORTED_IMAGES_FOLDER) if f.is_dir()]

# Collect class information
class_names = {}
class_id = 0

# build class_names dict first
for folder in folders:
    class_name = os.path.basename(folder)
    if class_name[0].isdigit():
        class_names[class_id] = class_name
        class_id += 1

# create reverse mapping
class_name_to_YOLOid = {v: k for k, v in class_names.items()}
class_id_to_YOLOid = {}
for class_name, yolo_id in class_name_to_YOLOid.items():
    class_id = class_name.split('_')[0]
    # print(f"Class '{class_id}' -> YOLO ID: {yolo_id}")
    class_id_to_YOLOid[class_id] = yolo_id
print(f"\nClass ID to YOLO ID mapping: {class_id_to_YOLOid}")

for folder in folders:
    class_name = os.path.basename(folder)
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
    if not class_name[0].isdigit():
    #     # Store class information
    #     class_names[class_id] = class_name
    #     class_id += 1
    # else:
    #     print(f"need to move '{class_name}' directly as it won't be added to class_id_to_YOLOid.")
        split_dataset(
            source_images= this_folder_images,
            source_labels=os.path.join(this_folder, 'labels'),
            output_dir=YOLO_READY_DATASET_FOLDER,
            class_id_to_YOLOid=class_id_to_YOLOid,
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
    yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

print(f"\nCreated YOLO config file: {yaml_path}")
print(f"Total classes: {len(class_names)}")
print(f"Classes: {class_names}")


for class_name, yolo_id in class_name_to_YOLOid.items():
    this_folder = os.path.join(SORTED_IMAGES_FOLDER,class_name)
    print(f"Processing folder: {this_folder}")
    this_folder_images = os.path.join(this_folder, 'images')

    split_dataset(
        source_images= this_folder_images,
        source_labels=os.path.join(this_folder, 'labels'),
        output_dir=YOLO_READY_DATASET_FOLDER,
        class_id_to_YOLOid=class_id_to_YOLOid,
        train_ratio=0.8
    )
