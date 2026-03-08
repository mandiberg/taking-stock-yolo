#!/usr/bin/env python3
"""Remap YOLO labels to COCO class IDs.

Maps local class IDs to COCO dataset class IDs based on class names.
"""
from __future__ import annotations

import argparse
from pathlib import Path

default_path = "/Users/michael.mandiberg/Documents/YOLO_Training_Data/sorted_images_guns/misc_guns_v10"
# default_path = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/YOLO_Training_Data/new_images/groceries_final_mixed"
# Mapping from local class ID to class name
# LOCAL_CLASS_NAMES = {
#     0: "Bag",
#     1: "Bitcoin",
#     2: "Creditcard",
#     3: "Dumbbell",
#     4: "Gift",
#     5: "Groceries",
#     6: "Iris",
#     7: "Lily",
#     8: "Lisianthus",
#     9: "Money",
#     10: "Orchid",
#     11: "Peony",
#     12: "Piggybank",
#     13: "Rose",
#     14: "Sign",
#     15: "Tulip",
# }



# LOCAL_CLASS_NAMES = {
#     0: "Daffodil",
#     1: "Daisy",
#     2: "Hydrangea",
#     3: "Iris",
#     4: "Lily",
#     5: "Lisianthus",
#     6: "Orchid",
#     7: "Peony",
#     8: "Rose",
#     9: "Sunflower",
#     10: "Tulip"
# }

LOCAL_CLASS_NAMES = {
    0: "Pistol",
    1: "Rifle",
}



# Mapping from class name to COCO class ID
COCO_CLASS_IDS = {
    "Sign": 80,
    "Gift": 81,
    "Money": 82,
    "Bag": 83,
    "Dumbbell": 86,
    "Groceries": 88,
    "Piggybank": 94,
    "Creditcard": 95,
    "Bitcoin": 96,
    "Rose": 97,
    "Lily": 98,
    "Iris": 99,
    "Tulip": 100,
    "Lisianthus": 101,
    "Orchid": 102,
    "Peony": 103,
    "Sunflower": 104,
    "Daisy": 105,
    "Daffodil": 106,
    "Hydrangea": 107,
    "Pistol": 108,
    "Rifle": 109,
}

# Build mapping from local class ID to COCO class ID
LOCAL_TO_COCO = {
    local_id: COCO_CLASS_IDS[LOCAL_CLASS_NAMES[local_id]]
    for local_id in LOCAL_CLASS_NAMES
    if LOCAL_CLASS_NAMES[local_id] in COCO_CLASS_IDS
}

def remap_labels(base_dir: Path) -> None:
    labels_dir = base_dir / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    total_files = 0
    changed_files = 0
    remapped_boxes = 0
    unmapped_boxes = 0

    for txt_path in labels_dir.rglob("*.txt"):
        total_files += 1
        with txt_path.open("r") as f:
            lines = f.read().strip().splitlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # skip malformed lines
            try:
                cls = int(float(parts[0]))
            except ValueError:
                continue

            if cls not in LOCAL_TO_COCO:
                unmapped_boxes += 1
                continue

            # Remap to COCO class ID
            parts[0] = str(LOCAL_TO_COCO[cls])
            remapped_boxes += 1
            new_lines.append(" ".join(parts))

        if new_lines != lines:
            changed_files += 1
            txt_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))

    print(f"Processed label files in: {labels_dir}")
    print(f"Files scanned: {total_files}")
    print(f"Files changed: {changed_files}")
    print(f"Boxes remapped: {remapped_boxes}")
    print(f"Boxes unmapped: {unmapped_boxes}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remap YOLO class IDs to COCO class IDs")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(default_path),
        help="Base dataset directory containing images/ and labels/",
    )

    args = parser.parse_args()
    remap_labels(args.base_dir)


if __name__ == "__main__":
    main()
