import os
from ultralytics import YOLO
import torch
import pandas as pd
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------------
# DEVICE SETUP (MPS on macOS GPU)
# -----------------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# -----------------------------------
# LOAD MODELS
# -----------------------------------
coco_model = YOLO("yolov8s.pt").to(device)
custom_model = YOLO("/Users/michael.mandiberg/Documents/GitHub/taking-stock-yolo/runs/takingstock_yolov8n13/weights/best.pt").to(device)

# -----------------------------------
# CONFIGURATION
# -----------------------------------
image_folder = "/Users/michael.mandiberg/Documents/takingstock_production/labeled_images_nov19/images_testing"
batch_size = 8               # Adjust based on memory
num_threads = 4
conf_thresh = 0.25

# -----------------------------------
# HELPER: RUN INFERENCE FOR ONE IMAGE
# -----------------------------------
def process_image(img_path):
    """Runs inference for one image with both models using OpenCV."""
    img_name = os.path.basename(img_path)

    try:
        # Load image using OpenCV (BGR format)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Failed to load image")

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        detections = []

        # ---- Run COCO model ----
        coco_results = coco_model.predict(
            img_rgb,
            conf=conf_thresh,
            device=device,
            verbose=False
        )[0]

        # ---- Run Custom model ----
        custom_results = custom_model.predict(
            img_rgb,
            conf=conf_thresh,
            device=device,
            verbose=False
        )[0]

        # --------------------------
        # COCO detections
        # --------------------------
        for b in coco_results.boxes:
            detections.append({
                "image": img_name,
                "model": "coco",
                "class_id": int(b.cls),
                "class_name": coco_results.names[int(b.cls)],
                "confidence": float(b.conf),
                "x1": float(b.xyxy[0][0]),
                "y1": float(b.xyxy[0][1]),
                "x2": float(b.xyxy[0][2]),
                "y2": float(b.xyxy[0][3]),
            })

        # --------------------------
        # Custom detections
        # --------------------------
        for b in custom_results.boxes:
            detections.append({
                "image": img_name,
                "model": "custom",
                "class_id": int(b.cls),
                "class_name": custom_results.names[int(b.cls)],
                "confidence": float(b.conf),
                "x1": float(b.xyxy[0][0]),
                "y1": float(b.xyxy[0][1]),
                "x2": float(b.xyxy[0][2]),
                "y2": float(b.xyxy[0][3]),
            })

        return detections

    except Exception as e:
        print(f"Error processing {img_name}: {e}")
        return []

# -----------------------------------
# GATHER IMAGE PATHS
# -----------------------------------
image_paths = [
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.lower().endswith(("jpg", "jpeg", "png", "bmp", "tif", "tiff"))
]

print(f"Found {len(image_paths)} images.")

# -----------------------------------
# BATCH + THREADING INFERENCE
# -----------------------------------
all_detections = []

def batch(lst, n):
    """Yield successive n-sized batches from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

for batch_paths in batch(image_paths, batch_size):
    with ThreadPoolExecutor(max_workers=num_threads) as exe:
        futures = {exe.submit(process_image, p): p for p in batch_paths}

        for fut in as_completed(futures):
            all_detections.extend(fut.result())

# -----------------------------------
# CREATE DATAFRAME
# -----------------------------------
df = pd.DataFrame(all_detections)

print("\n🔎 FINAL RESULTS DATAFRAME:")
print(df.head(50))
print(f"\nTotal detections: {len(df)}")

# Optional: save to CSV
df.to_csv("detections.csv", index=False)
print("\nSaved detections.csv")


# display each image with boxes (optional)for img_name in df['image'].unique():
for img_name in df['image'].unique():   
    img_path = os.path.join(image_folder, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_detections = df[df['image'] == img_name]
    for _, row in img_detections.iterrows():
        x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
        label = f"{row['model']}:{row['class_name']} {row['confidence']:.2f}"
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Detections", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
cv2.destroyAllWindows()