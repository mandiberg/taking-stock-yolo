import os

ROOT = "/Users/michael.mandiberg/Documents/YOLO_Training_Data/sorted_images"
# SPLITS = ["train", "val"]
folders = [f.path for f in os.scandir(ROOT) if f.is_dir()]
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

def delete_pair(label_path, image_dir):
    base = os.path.splitext(os.path.basename(label_path))[0]

    # delete label
    if os.path.exists(label_path):
        os.remove(label_path)
        print(f"🗑️ deleted label: {label_path}")

    # delete corresponding image (any extension)
    for ext in IMAGE_EXTS:
        img_path = os.path.join(image_dir, base + ext)
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"🗑️ deleted image: {img_path}")
            return

    print(f"⚠️ image not found for label {base}")

for folder in folders:
    label_dir = os.path.join(folder, "labels")
    image_dir = os.path.join(folder, "images")

    if not os.path.isdir(label_dir):
        print(f"Labels directory not found: {label_dir}")
        exit(1)

    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue

        label_path = os.path.join(label_dir, fname)

        with open(label_path) as f:
            lines = f.readlines()

        invalid = False

        # if len(lines) == 0:
        #     invalid = True  # empty label file

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                invalid = True
                break

            try:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                invalid = True
                break

            # basic YOLO validity checks
            if cls < 0:
                invalid = True
                break
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                invalid = True
                break
            if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                invalid = True
                break

        if invalid:
            delete_pair(label_path, image_dir)
