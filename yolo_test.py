from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/detect/money_detector/weights/best.pt')

# Test on new image
results = model('/Users/michaelmandiberg/Documents/GitHub/facemap/yolo_dataset/images/val/7db1e25a-1963428697.jpg')

# Show results
results[0].show()

# Or get detailed info
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Class: {r.names[cls]}, Confidence: {conf:.2f}")