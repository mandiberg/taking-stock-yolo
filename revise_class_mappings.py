#!/usr/bin/env python3
"""Remap YOLO labels for the HF flowers dataset.

- Drops labels with class id > max_class (default 1).
- Maps class 1 -> 0 (combining "flower arrangement" with "flower").
- Leaves class 0 unchanged.

Use after dataset download/conversion to prune irrelevant classes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

default_path = "./hf_yolo_dataset"
max_class = 1

def remap_labels(base_dir: Path, max_class: int = 1) -> None:
    labels_dir = base_dir / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    total_files = 0
    changed_files = 0
    dropped_boxes = 0
    kept_boxes = 0

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

            if cls > max_class:
                dropped_boxes += 1
                continue

            if cls == 1:
                parts[0] = "0"  # map class 1 -> 0
            # cls == 0 stays 0
            kept_boxes += 1
            new_lines.append(" ".join(parts))

        if new_lines != lines:
            changed_files += 1
            txt_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))

    print(f"Processed label files in: {labels_dir}")
    print(f"Files scanned: {total_files}")
    print(f"Files changed: {changed_files}")
    print(f"Boxes kept:   {kept_boxes}")
    print(f"Boxes dropped: {dropped_boxes}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remap YOLO class ids for flowers dataset")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(default_path),
        help="Base dataset directory containing images/ and labels/",
    )
    parser.add_argument(
        "--max-class",
        type=int,
        default=1,
        help="Keep classes <= max_class; drop the rest",
    )

    args = parser.parse_args()
    remap_labels(args.base_dir, max_class=args.max_class)


if __name__ == "__main__":
    main()
