#!/usr/bin/env python3
"""Remap YOLO labels to COCO class IDs.

Maps local class IDs to COCO dataset class IDs based on class names.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from class_map_utils import resolve_class_name_to_id

default_path = "/Users/michaelmandiberg/Documents/yolo/reprocess/calculator_good"
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

# LOCAL_CLASS_NAMES = {
#     0: "Avocado_half",
#     1: "Chestpiece",
#     2: "Cucumber",
#     3: "Eyepatch",
#     4: "Facial",
#     5: "Headphones",
#     6: "Kiwi",
#     7: "Lemon_slice",
#     8: "Mask",
#     9: "Masquerade_mask",
#     10: "Sheetmask",
#     11: "Sleepmask",
#     12: "Stethoscope",
#     13: "Valentine"
# }  

LOCAL_CLASS_NAMES = {
    0: "Avocado half",
    1: "Bag",
    2: "Bitcoin",
    3: "Boxing gloves",
    4: "Chestpiece",
    5: "Cigarette",
    6: "Clipboard",
    7: "Creditcard",
    8: "Cucumber",
    9: "Daffodil",
    10: "Daisy",
    11: "Dumbbell",
    12: "Eyeglasses",
    13: "Eyepatch",
    14: "Facial",
    15: "Flag",
    16: "Gift",
    17: "Groceries",
    18: "Gun",
    19: "Headphones",
    20: "Hydrangea",
    21: "Iris",
    22: "Kiwi",
    23: "Lemon slice",
    24: "Lily",
    25: "Lisianthus",
    26: "Mask",
    27: "Masquerade mask",
    28: "Money",
    29: "Orchid",
    30: "Peony",
    31: "Picture frame",
    32: "Piggybank",
    33: "Pistol",
    34: "Playing cards",
    35: "Rifle",
    36: "Rose",
    37: "Salad",
    38: "Sheetmask",
    39: "Sign",
    40: "Sleepmask",
    41: "Stethoscope",
    42: "Sunflower",
    43: "Tablet",
    44: "Tulip",
    45: "Valentine",
    46: "Vape",
    47: "Calculator",
}

LOCAL_TO_COCO = {}
UNMAPPED_LOCAL = {}
for local_id, local_name in LOCAL_CLASS_NAMES.items():
    resolved = resolve_class_name_to_id(local_name)
    if resolved is None:
        UNMAPPED_LOCAL[local_id] = local_name
        continue
    LOCAL_TO_COCO[local_id] = resolved

def remap_labels(base_dir: Path) -> None:
    labels_dir = base_dir / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    if UNMAPPED_LOCAL:
        print("Warning: some LOCAL_CLASS_NAMES entries are not in custom_class_map.json")
        for local_id, local_name in sorted(UNMAPPED_LOCAL.items()):
            print(f"  local {local_id}: {local_name}")

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
