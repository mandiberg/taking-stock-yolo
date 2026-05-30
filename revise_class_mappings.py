#!/usr/bin/env python3
"""Remap YOLO labels to COCO class IDs.

Maps local class IDs to COCO dataset class IDs based on class names.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from class_map_utils import resolve_class_name_to_id

YOLO_ROOT = "/Users/michaelmandiberg/Documents/YOLO_Training_Data/Relabeled_johnmarotta_april2026/"
WALK_FOLDERS = True
SUBFOLDER_PATH = None

# YOLO_ROOT = "/Users/michaelmandiberg/Documents/yolo/reprocess/relabel_these93"
# YOLO_ROOT = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/YOLO_Training_Data/new_images/groceries_final_mixed"
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
    46: "Vape"
}



# LOCAL_CLASS_NAMES = {

#     0: "Avocado_half",
#     1: "Bag",
#     2: "Binoculars",
#     3: "Bitcoin",
#     4: "Book_custom",
#     5: "Boxing_gloves",
#     6: "Calculator",
#     7: "Camera",
#     8: "Chestpiece",
#     9: "Cigarette",
#     10: "Clipboard",
#     11: "Computer_monitor",
#     12: "Corded_phone",
#     13: "Creditcard",
#     14: "Cucumber",
#     15: "Daffodil",
#     16: "Daisy",
#     17: "Drill_tool",
#     18: "Dumbbell",
#     19: "Eyepatch",
#     20: "Facial",
#     21: "Flag",
#     22: "Gift",
#     23: "Glasses",
#     24: "Groceries",
#     25: "Headphones",
#     26: "Hydrangea",
#     27: "Iris",
#     28: "Keybaord_custom",
#     29: "Kiwi",
#     30: "Laptop_custom",
#     31: "Lemon_slice",
#     32: "Lily",
#     33: "Lisianthus",
#     34: "Mask",
#     35: "Masquerade_mask",
#     36: "Megaphone",
#     37: "Microphone",
#     38: "Money",
#     39: "None",
#     40: "Orchid",
#     41: "Peony",
#     42: "Phone_handheld",
#     43: "Picture_frame",
#     44: "Piggybank",
#     45: "Pistol",
#     46: "Playing_cards",
#     47: "Remote_control_custom",
#     48: "Rifle",
#     49: "Rose",
#     50: "Salad",
#     51: "Sheetmask",
#     52: "Sign",
#     53: "Sleepmask",
#     54: "Stethoscope",
#     55: "Sunflower",
#     56: "Tablet",
#     57: "Tulip",
#     58: "Valentine",
#     59: "Vape" 
 
# }

LOCAL_TO_COCO = {}
UNMAPPED_LOCAL = {}
for local_id, local_name in LOCAL_CLASS_NAMES.items():
    resolved = resolve_class_name_to_id(local_name)
    if resolved is None:
        UNMAPPED_LOCAL[local_id] = local_name
        continue
    LOCAL_TO_COCO[local_id] = resolved


def is_hidden_name(name: str) -> bool:
    return name.startswith(".")


def discover_dataset_folders(yolo_root: Path) -> list[Path]:
    dataset_folders: list[Path] = []
    for name in sorted(os.listdir(yolo_root)):
        if is_hidden_name(name):
            continue

        dataset_path = yolo_root / name
        if not dataset_path.is_dir():
            continue

        target_base = dataset_path / SUBFOLDER_PATH if SUBFOLDER_PATH else dataset_path
        labels_dir = target_base / "labels"
        if labels_dir.is_dir():
            dataset_folders.append(target_base)
        else:
            print(f"Skipping {dataset_path}: missing labels at {labels_dir}")

    return dataset_folders


def remap_labels(base_dir: Path) -> dict[str, int]:
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

    return {
        "files": total_files,
        "changed": changed_files,
        "remapped": remapped_boxes,
        "unmapped": unmapped_boxes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Remap YOLO class IDs to COCO class IDs")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(YOLO_ROOT),
        help="Base dataset directory containing images/ and labels/",
    )

    args = parser.parse_args()

    if WALK_FOLDERS:
        yolo_root = args.base_dir
        if not yolo_root.is_dir():
            raise FileNotFoundError(f"YOLO_ROOT not found: {yolo_root}")

        dataset_bases = discover_dataset_folders(yolo_root)
        if not dataset_bases:
            print("No valid dataset folders found.")
            return

        totals = {"files": 0, "changed": 0, "remapped": 0, "unmapped": 0}
        for dataset_base in dataset_bases:
            print(f"\n=== Processing dataset path: {dataset_base} ===")
            stats = remap_labels(dataset_base)
            for key in totals:
                totals[key] += stats[key]

        print("\nSummary:")
        print(f"Datasets processed: {len(dataset_bases)}")
        print(f"Files scanned: {totals['files']}")
        print(f"Files changed: {totals['changed']}")
        print(f"Boxes remapped: {totals['remapped']}")
        print(f"Boxes unmapped: {totals['unmapped']}")
    else:
        remap_labels(args.base_dir)


if __name__ == "__main__":
    main()
