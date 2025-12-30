from ultralytics import YOLO
from datasets import load_dataset
import os
from pathlib import Path
import yaml
from PIL import Image
import requests
from io import BytesIO

# Load dataset from Hugging Face
print("Loading dataset from Hugging Face...")
ds = load_dataset("visual-layer/oxford-flowers-vl-enriched")

print(f"Dataset loaded with {len(ds['train'])} training examples")

# Create dataset directory structure
base_dir = Path("./hf_yolo_dataset")
yaml_path = base_dir / 'data.yaml'
dataset_prepared = yaml_path.exists() and (base_dir / 'images' / 'train').exists()

# Check if dataset is already prepared
if dataset_prepared:
    print(f"\nDataset already prepared at {base_dir}")
    print("Skipping download and conversion, using existing data...")
    with open(yaml_path, 'r') as f:
        data_yaml = yaml.safe_load(f)
    class_names = data_yaml.get('names', {})
else:
    print("\nPreparing dataset (this may take a few minutes)...")
    base_dir.mkdir(exist_ok=True)
    for split in ['train', 'test']:
        (base_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (base_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

# Function to convert dataset to YOLO format
def convert_to_yolo_format(dataset_split, split_name, label_to_id):
    print(f"Converting {split_name} split...")
    
    next_id = max(label_to_id.values()) + 1 if label_to_id else 0
    
    for idx, item in enumerate(dataset_split):
        if idx % 500 == 0:
            print(f"  Processing image {idx}/{len(dataset_split)}...")
        
        try:
            # Download image from URL
            response = requests.get(item['image_uri'], timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save image
            image_filename = f"{split_name}_{idx:05d}.jpg"
            image_path = base_dir / 'images' / split_name / image_filename
            image.save(image_path, 'JPEG')
            
            # Get image dimensions
            img_width, img_height = image.size
            
            # Convert annotations to YOLO format
            label_path = base_dir / 'labels' / split_name / f"{split_name}_{idx:05d}.txt"
            
            with open(label_path, 'w') as f:
                # Process object labels if available
                if 'object_labels' in item and item['object_labels']:
                    for obj in item['object_labels']:
                        label = obj.get('label', 'object')
                        bbox = obj.get('bbox', None)
                        
                        # Map label to ID (use existing or create new)
                        if label not in label_to_id:
                            label_to_id[label] = next_id
                            next_id += 1
                        class_id = label_to_id[label]
                        
                        if bbox and len(bbox) == 4:
                            # bbox format: [x_min, y_min, x_max, y_max]
                            x_min, y_min, x_max, y_max = bbox
                            
                            # Convert to YOLO format (normalized x_center, y_center, width, height)
                            x_center = ((x_min + x_max) / 2) / img_width
                            y_center = ((y_min + y_max) / 2) / img_height
                            width = (x_max - x_min) / img_width
                            height = (y_max - y_min) / img_height
                            
                            # Ensure values are within [0, 1]
                            x_center = max(0, min(1, x_center))
                            y_center = max(0, min(1, y_center))
                            width = max(0, min(1, width))
                            height = max(0, min(1, height))
                            
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                # If no object labels, use image_label for full-image classification
                elif 'image_label' in item:
                    label = f"class_{item['image_label']}"
                    if label not in label_to_id:
                        label_to_id[label] = next_id
                        next_id += 1
                    class_id = label_to_id[label]
                    # Full image box in YOLO format
                    f.write(f"{class_id} 0.5 0.5 0.99 0.99\n")
        
        except Exception as e:
            print(f"  Warning: Failed to process image {idx}: {e}")
            continue
    
    print(f"  Completed {split_name} split.")
    return label_to_id

if not dataset_prepared:
    # Convert train split and collect class labels
    label_to_id = {}
    if 'train' in ds:
        label_to_id = convert_to_yolo_format(ds['train'], 'train', label_to_id)

    # For validation, we'll use a portion of train or test split if available
    if 'test' in ds:
        label_to_id = convert_to_yolo_format(ds['test'], 'test', label_to_id)
    elif len(ds['train']) > 100:
        # Use last 10% of train as validation
        val_size = len(ds['train']) // 10
        val_split = ds['train'].select(range(len(ds['train']) - val_size, len(ds['train'])))
        label_to_id = convert_to_yolo_format(val_split, 'test', label_to_id)

    # Create class names dictionary
    class_names = {v: k for k, v in label_to_id.items()}

    # Create data.yaml
    data_yaml = {
        'path': str(base_dir.absolute()),
        'train': 'images/train',
        'val': 'images/test',
        'nc': len(class_names),
        'names': class_names
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"Dataset prepared at {base_dir}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")

# Load pretrained model
model = YOLO('yolov8m.pt')  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt

# Train
print("\nStarting training...")
results = model.train(
    data=str(yaml_path.absolute()),
    epochs=100,
    imgsz=640,
    batch=16,       # Reduce if you get memory errors
    name='flowers_detector',
    patience=20,    # Early stopping
    device='cpu',   # Use GPU, or 'cpu' for CPU only
)