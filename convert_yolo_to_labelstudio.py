import os
import json
import uuid
import cv2

# 📌 CONFIG — adjust these
YOLO_ROOT = "/Volumes/Michael Mandiberg’s Public Folder/steth_chestpiece"
IMAGES_DIR = os.path.join(YOLO_ROOT, "images")
LABELS_DIR = os.path.join(YOLO_ROOT, "labels")
OUTPUT_JSON = os.path.join(YOLO_ROOT, "labelstudio_tasks.json")

# 📌 You must manually list your classes here
CLASSES = [
    "class0", "class1", "class2",  # replace with your actual class names
]

tasks = []

print(f"Processing YOLO dataset at:\n  {YOLO_ROOT}")

for img_name in sorted(os.listdir(IMAGES_DIR)):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(IMAGES_DIR, img_name)

    # load image to get dimensions
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not read image {img_name}, skipping.")
        continue

    h, w = img.shape[:2]

    # find YOLO label
    base = os.path.splitext(img_name)[0]
    label_txt = os.path.join(LABELS_DIR, base + ".txt")
    if not os.path.exists(label_txt):
        print(f"No label file found for {img_name}, skipping.")
        continue

    results = []

    with open(label_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Skipping malformed line in {label_txt}: {line.strip()}")
                continue

            class_id, xc, yc, bw, bh = parts
            class_id = int(class_id)
            xc, yc, bw, bh = map(float, [xc, yc, bw, bh])

            # convert YOLO normalized center to left-top %
            left = (xc - bw/2) * 100
            top  = (yc - bh/2) * 100
            width  = bw * 100
            height = bh * 100

            label_name = CLASSES[class_id] if class_id < len(CLASSES) else f"class{class_id}"

            # unique ID for this box
            uid = str(uuid.uuid4())

            result = {
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": left,
                    "y": top,
                    "width": width,
                    "height": height,
                    "rectanglelabels": [label_name]
                },
                "id": uid,
                "score": 1.0
            }
            results.append(result)

    if not results:
        print(f"No boxes for {img_name}, skipping.")
        continue

    task = {
        "data": {
            "image": img_name  # refer to image by its filename
        },
        "predictions": [
            {
                "result": results
            }
        ]
    }

    tasks.append(task)
    print(f"✔ Converted {img_name} with {len(results)} boxes")

# write output file
with open(OUTPUT_JSON, "w") as f:
    json.dump(tasks, f, indent=2)

print(f"\nDone! Label Studio tasks written to:\n  {OUTPUT_JSON}")
