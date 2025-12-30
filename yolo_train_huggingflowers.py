from ultralytics import YOLO
from datasets import load_dataset
import os
from pathlib import Path
import yaml
from PIL import Image

# Load dataset from Hugging Face
print("Loading dataset from Hugging Face...")
ds = load_dataset("visual-layer/oxford-flowers-vl-enriched")

# Create dataset directory structure
base_dir = Path("./hf_yolo_dataset")
base_dir.mkdir(exist_ok=True)

for split in ['train', 'test']:
    (base_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
    (base_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

# Function to convert dataset to YOLO format
def convert_to_yolo_format(dataset_split, split_name):
    print(f"Converting {split_name} split...")
    
    for idx, item in enumerate(dataset_split):
        # Get image
        image = item['image']
        
        # Save image
        image_filename = f"{split_name}_{idx:05d}.jpg"
        image_path = base_dir / 'images' / split_name / image_filename
        image.save(image_path)
        
        # Convert annotations to YOLO format if available
        label_path = base_dir / 'labels' / split_name / f"{split_name}_{idx:05d}.txt"
        
        # Check if the dataset has bounding box annotations
        if 'objects' in item and item['objects']:
            with open(label_path, 'w') as f:
                # Get image dimensions
                img_width, img_height = image.size
                
                objects = item['objects']
                # Handle both dict and list of dicts format
                if isinstance(objects, dict):
                    bboxes = objects.get('bbox', [])
                    categories = objects.get('category', [])
                    
                    for bbox, category in zip(bboxes, categories):
                        # bbox format: [x_min, y_min, x_max, y_max]
                        x_min, y_min, x_max, y_max = bbox
                        
                        # Convert to YOLO format (normalized x_center, y_center, width, height)
                        x_center = ((x_min + x_max) / 2) / img_width
                        y_center = ((y_min + y_max) / 2) / img_height
                        width = (x_max - x_min) / img_width
                        height = (y_max - y_min) / img_height
                        
                        f.write(f"{category} {x_center} {y_center} {width} {height}\n")
        else:
            # If no bounding boxes, check for classification labels
            if 'label' in item:
                # For classification, create a full image bounding box
                with open(label_path, 'w') as f:
                    # Full image box in YOLO format (center at 0.5, 0.5, full width/height)
                    f.write(f"{item['label']} 0.5 0.5 0.99 0.99\n")

# Convert train and test splits
if 'train' in ds:
    convert_to_yolo_format(ds['train'], 'train')
if 'test' in ds:
    convert_to_yolo_format(ds['test'], 'test')
elif 'validation' in ds:
    convert_to_yolo_format(ds['validation'], 'test')

# Get class names from the dataset
# Try to extract class names from dataset features
class_names = {}
if 'train' in ds:
    sample = ds['train'][0]
    
    # Try to get class names from the dataset features
    if hasattr(ds['train'].features.get('label', None), 'names'):
        names_list = ds['train'].features['label'].names
        class_names = {i: name for i, name in enumerate(names_list)}
    elif 'objects' in sample and isinstance(sample['objects'], dict):
        if hasattr(ds['train'].features['objects'].feature.get('category', None), 'names'):
            names_list = ds['train'].features['objects'].feature['category'].names
            class_names = {i: name for i, name in enumerate(names_list)}

# If no class names found, create generic ones
if not class_names:
    class_names = {0: 'object'}

# Create data.yaml
data_yaml = {
    'path': str(base_dir.absolute()),
    'train': 'images/train',
    'val': 'images/test',
    'nc': len(class_names),
    'names': class_names
}

yaml_path = base_dir / 'data.yaml'
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