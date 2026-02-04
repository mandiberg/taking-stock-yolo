from ultralytics import YOLO

# Load your trained model
model = YOLO('/Users/michael.mandiberg/Documents/GitHub/takingstock/models/takingstock_84_valentine_withblems_yolov8m/weights/best.pt')

# Test on new image
results = model('/Volumes/LaCie/segment_images_84_valentine/test_output_noblems/84/unique/0.98_35214354_YOLO_debug.jpg')

# Show results
results[0].show()

# Or get detailed info
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Class: {r.names[cls]}, Confidence: {conf:.2f}")