import os
import json
from pathlib import Path

# ---------------- CONFIG ----------------

IMAGES_DIR = Path("/Users/michaelmandiberg/Downloads/Project_Image_Ann/images")
LABELS_DIR = Path("/Users/michaelmandiberg/Downloads/Project_Image_Ann/labels")
OUTPUT_JSON = "labelstudio_import.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

LABEL_NAMES = [
    "ATM Card", "Bag", "Bitcoin", "Creditcard", "Dumbbell",
    "Gift", "Groceries", "Iris", "Keyboard", "Lily",
    "Lisianthus", "Mic", "Mobile", "Money", "Orchid",
    "Peony", "Piggybank", "Rose", "Sign", "Tulip"
]

# ---------------------------------------


def yolo_to_ls_bbox(class_id, xc, yc, w, h):
    """
    Convert YOLO normalized center coords to Label Studio percent top-left coords
    """
    return {
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "value": {
            "x": (xc - w / 2) * 100,
            "y": (yc - h / 2) * 100,
            "width": w * 100,
            "height": h * 100,
            "rectanglelabels": [LABEL_NAMES[class_id]],
        },
    }


def find_label_file(image_stem):
    """
    Prefer exact match: image.jpg -> image.txt
    Fallback: UID-image.txt
    """
    exact = LABELS_DIR / f"{image_stem}.txt"
    if exact.exists():
        return exact

    # fallback: *-image_stem.txt
    candidates = list(LABELS_DIR.glob(f"*-{image_stem}.txt"))
    if candidates:
        return candidates[0]

    return None


tasks = []

for image_path in sorted(IMAGES_DIR.iterdir()):
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    image_stem = image_path.stem
    label_file = find_label_file(image_stem)

    results = []

    if label_file and label_file.exists():
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])

                results.append(
                    yolo_to_ls_bbox(class_id, xc, yc, w, h)
                )

    task = {
        "data": {
            "image": f"file://{image_path.resolve()}"
        },
        "annotations": [
            {
                "result": results
            }
        ]
    }

    tasks.append(task)

print(f"Created {len(tasks)} tasks")

with open(OUTPUT_JSON, "w") as f:
    json.dump(tasks, f, indent=2)

print(f"Wrote {OUTPUT_JSON}")
