from ultralytics import YOLO

# Load your trained model
model = YOLO('/Users/michael.mandiberg/Documents/GitHub/taking-stock-yolo/runs/detect/takingstock_yolov8n/weights/best.pt')

# Test on new image
results = model('/Users/michael.mandiberg/Documents/GitHub/taking-stock-yolo/yolo_dataset/images/val/7db1e25a-1963428697.jpg')

# Show results
results[0].show()

# Or get detailed info
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Class: {r.names[cls]}, Confidence: {conf:.2f}")