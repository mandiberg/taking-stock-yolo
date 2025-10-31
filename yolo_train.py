from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt

# Train
results = model.train(
    data='/Users/michaelmandiberg/Documents/GitHub/facemap/yolo_dataset/data.yaml',  # absolute path
    epochs=100,
    imgsz=640,
    batch=16,       # Reduce if you get memory errors
    name='money_detector',
    patience=20,    # Early stopping
    device='cpu',       # Use GPU, or 'cpu' for CPU only
)