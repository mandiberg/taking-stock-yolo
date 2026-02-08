import os
import json
import uuid
import cv2

# 📌 CONFIG — adjust these
YOLO_ROOT = "/Users/michaelmandiberg/Documents/yolo/"
DATASET_FOLDER = "steth_chestpiece"
IMAGES_DIR = os.path.join(YOLO_ROOT, DATASET_FOLDER, "images")
LABELS_DIR = os.path.join(YOLO_ROOT, DATASET_FOLDER, "labels")
OUTPUT_JSON = os.path.join(YOLO_ROOT, DATASET_FOLDER, "labelstudio_tasks.json")

# 📌 You must manually list your classes here
CLASSES = {
    80: "Sign",
    81: "Gift",
    82: "Money",
    83: "Bag",
    84: "Valentine",
    85: "Salad",
    86: "Dumbbell",
    87: "Flag",
    88: "Groceries",
    89: "Mask",
    90: "Stethoscope",
    91: "Gun",
    92: "Headphones",
    93: "Clipboard",
    94: "Piggybank",
    95: "Creditcard",
    96: "Bitcoin",
    97: "Rose",
    98: "Lily",
    99: "Iris",
    100: "Tulip",
    101: "Lisianthus",
    102: "Orchid",
    103: "Peony",

}


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

            print(f"Converting box for class ID {class_id} in {img_name} with {min(CLASSES.keys())} classes.")
            label_name = CLASSES[class_id] if class_id >= min(CLASSES.keys()) else f"class{class_id}"
            print(f" - Found box: {label_name} at ({left:.2f}%, {top:.2f}%, {width:.2f}%, {height:.2f}%)")
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
                }
                # ,
                # "id": uid,
                # "score": 1.0
            }
            results.append(result)

    if not results:
        print(f"No boxes for {img_name}, skipping.")
        continue

    task = {
        "data": {
            "image": img_name  # refer to image by its filename
        },
        "detections": [
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
