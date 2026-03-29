#!/usr/bin/env python3
"""Convert MFSD segmentation masks to YOLOv8 bbox labels.

This script reads MFSD mask images from:
- <MSFD_PATH>/1/face_crop_segmentation
and matches them with source images in:
- <MSFD_PATH>/1/face_crop

For each segmentation mask, it computes a single bounding box around non-zero pixels
(mask region) and writes YOLO label files.

Important:
- This script never edits files in the source dataset.
- Output is written to a separate folder only.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

DEFAULT_MSFD_PATH = Path("/Users/michaelmandiberg/Documents/yolo/MSFD")
DEFAULT_OUTPUT_FOLDER = Path("/Users/michaelmandiberg/Documents/yolo/MSFD_YOLO")
DEFAULT_TIGHTNESS = 0
STRAP_TRIM_PERCENTILE = .2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MFSD segmentation masks into YOLOv8 bounding-box labels."
    )
    parser.add_argument(
        "--msfd-path",
        type=Path,
        default=DEFAULT_MSFD_PATH,
        help=f"Path to MFSD root directory (default: {DEFAULT_MSFD_PATH})",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=DEFAULT_OUTPUT_FOLDER,
        help=f"Output directory for YOLO dataset (default: {DEFAULT_OUTPUT_FOLDER})",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="YOLO class ID for mask bbox labels (default: 0)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Grayscale threshold for mask foreground (0-255, default: 10)",
    )
    parser.add_argument(
        "--tightness",
        type=float,
        default=DEFAULT_TIGHTNESS,
        help=(
            "Shrink bbox around the detected mask fabric (0.0-0.5). "
            f"Higher = tighter box (default: {DEFAULT_TIGHTNESS})"
        ),
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Process only a small subset for quick testing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max files to process when --test-mode is enabled (default: 100)",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        default=True,
        help="Copy source images into output/images (default: enabled).",
    )
    parser.add_argument(
        "--no-copy-images",
        dest="copy_images",
        action="store_false",
        help="Do not copy images; only write labels.",
    )
    parser.add_argument(
        "--write-data-yaml",
        action="store_true",
        default=True,
        help="Write a basic YOLO data.yaml file in output root (default: enabled).",
    )
    parser.add_argument(
        "--no-write-data-yaml",
        dest="write_data_yaml",
        action="store_false",
        help="Skip writing data.yaml.",
    )
    return parser.parse_args()


def find_image_files(folder: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            files[path.stem] = path
    return files


def iter_pairs(
    source_images: dict[str, Path],
    segmentation_images: dict[str, Path],
) -> Iterable[tuple[str, Path, Path]]:
    common_stems = sorted(set(source_images.keys()) & set(segmentation_images.keys()))
    for stem in common_stems:
        yield stem, source_images[stem], segmentation_images[stem]


def mask_to_bbox(mask_path: Path, threshold: int) -> tuple[int, int, int, int] | None:
    with Image.open(mask_path) as mask_img:
        gray = np.array(mask_img.convert("L"), dtype=np.uint8)

    binary = gray > threshold
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return None

    # Remove thin outlier tails (typically elastic straps) using robust percentiles.
    x_low = int(np.floor(np.percentile(xs, STRAP_TRIM_PERCENTILE)))
    x_high = int(np.ceil(np.percentile(xs, 100.0 - STRAP_TRIM_PERCENTILE)))
    y_low = int(np.floor(np.percentile(ys, STRAP_TRIM_PERCENTILE)))
    y_high = int(np.ceil(np.percentile(ys, 100.0 - STRAP_TRIM_PERCENTILE)))

    x_low = max(0, x_low)
    y_low = max(0, y_low)
    x_high = min(gray.shape[1] - 1, x_high)
    y_high = min(gray.shape[0] - 1, y_high)

    if x_high <= x_low or y_high <= y_low:
        return None

    # PIL-style bbox uses exclusive right/bottom edges.
    return (x_low, y_low, x_high + 1, y_high + 1)


def tighten_bbox(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    tightness: float,
) -> tuple[int, int, int, int]:
    tightness = min(max(tightness, 0.0), 0.5)

    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top

    if width <= 2 or height <= 2 or tightness <= 0.0:
        return bbox

    shrink_x = int(round(width * tightness / 2.0))
    shrink_y = int(round(height * tightness / 2.0))

    new_left = left + shrink_x
    new_right = right - shrink_x
    new_top = top + shrink_y
    new_bottom = bottom - shrink_y

    # Keep bbox valid and inside image bounds.
    new_left = max(0, min(new_left, image_width - 2))
    new_top = max(0, min(new_top, image_height - 2))
    new_right = max(new_left + 1, min(new_right, image_width))
    new_bottom = max(new_top + 1, min(new_bottom, image_height))

    return (new_left, new_top, new_right, new_bottom)


def bbox_to_yolo(
    bbox: tuple[int, int, int, int], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    left, top, right, bottom = bbox

    box_w = max(0, right - left)
    box_h = max(0, bottom - top)
    center_x = left + box_w / 2.0
    center_y = top + box_h / 2.0

    x = center_x / image_width
    y = center_y / image_height
    w = box_w / image_width
    h = box_h / image_height

    # Clamp values in case of edge artifacts.
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)

    return x, y, w, h


def write_data_yaml(output_folder: Path) -> None:
    data_yaml = output_folder / "data.yaml"
    content = """path: .
train: images
val: images
names:
  0: mask
"""
    data_yaml.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()

    split1 = args.msfd_path / "1"
    source_dir = split1 / "face_crop"
    segmentation_dir = split1 / "face_crop_segmentation"

    if not source_dir.exists():
        raise FileNotFoundError(f"Source images folder not found: {source_dir}")
    if not segmentation_dir.exists():
        raise FileNotFoundError(f"Segmentation folder not found: {segmentation_dir}")

    out_images = args.output_folder / "images"
    out_labels = args.output_folder / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    source_images = find_image_files(source_dir)
    segmentation_images = find_image_files(segmentation_dir)

    pairs = list(iter_pairs(source_images, segmentation_images))
    if args.test_mode:
        pairs = pairs[: args.limit]

    print(f"Found {len(source_images)} source images in: {source_dir}")
    print(f"Found {len(segmentation_images)} segmentation images in: {segmentation_dir}")
    print(f"Matched {len(pairs)} image/mask pairs for processing")
    print(f"Output folder: {args.output_folder}")
    print("Source dataset remains unchanged (read-only access).")

    processed = 0
    written = 0
    empty = 0

    for stem, image_path, mask_path in pairs:
        processed += 1

        with Image.open(image_path) as img:
            img_w, img_h = img.size

        bbox = mask_to_bbox(mask_path, threshold=args.threshold)
        label_path = out_labels / f"{stem}.txt"

        if bbox is None:
            label_path.write_text("", encoding="utf-8")
            empty += 1
        else:
            bbox = tighten_bbox(
                bbox,
                image_width=img_w,
                image_height=img_h,
                tightness=args.tightness,
            )
            x, y, w, h = bbox_to_yolo(bbox, image_width=img_w, image_height=img_h)
            label_line = f"{args.class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
            label_path.write_text(label_line, encoding="utf-8")
            written += 1

        if args.copy_images:
            shutil.copy2(image_path, out_images / image_path.name)

        if processed % 100 == 0:
            print(
                f"Processed {processed}/{len(pairs)} | labels: {written} | empty: {empty}"
            )

    if args.write_data_yaml:
        write_data_yaml(args.output_folder)

    print("\nDone.")
    print(f"Processed: {processed}")
    print(f"Labels with bbox: {written}")
    print(f"Empty labels (no mask pixels found): {empty}")
    print(f"Saved labels to: {out_labels}")
    if args.copy_images:
        print(f"Saved images to: {out_images}")
    else:
        print("Image copying disabled (--no-copy-images).")


if __name__ == "__main__":
    main()
