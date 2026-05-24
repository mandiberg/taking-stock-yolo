#!/usr/bin/env python3
"""Convert eyeglasses segmentation files into YOLOv8 bbox labels.

Expected input layout:
- A single folder containing files with stems ending in variants like
    "-all", "-seg", and "-sunglasses".

For each base stem, this script requires all three files:
- <base>-all.<ext>
- <base>-seg.<ext>
- <base>-sunglasses.<ext>

The bounding box is computed from the foreground of the "seg" image and then
written to both "all" and "sunglasses" label files. Incomplete groups are skipped.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

DEFAULT_INPUT_FOLDER = Path("/Users/michaelmandiberg/Documents/yolo/glasses-segmentation-synthetic/test")
DEFAULT_OUTPUT_FOLDER = Path("/Users/michaelmandiberg/Documents/yolo/glasses-segmentation-synthetic/yolo")
DEFAULT_CLASS_ID = 120
DEFAULT_TIGHTNESS = 0
TARGET_VARIANTS = ("all", "seg", "sunglasses")


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert eyeglasses segmentation files into YOLOv8 bounding-box labels."
    )
    parser.add_argument(
        "--input-folder",
        type=Path,
        default=DEFAULT_INPUT_FOLDER,
        help=f"Path to folder containing all/seg/sunglasses files (default: {DEFAULT_INPUT_FOLDER})",
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
        default=DEFAULT_CLASS_ID,
        help=f"YOLO class ID for glasses bbox labels (default: {DEFAULT_CLASS_ID})",
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


class GroupFiles(NamedTuple):
    all_image: Path
    seg_image: Path
    sunglasses_image: Path


def find_image_files(folder: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            files[path.stem] = path
    return files


def split_variant(stem: str) -> tuple[str, str] | None:
    if "-" not in stem:
        return None
    base, variant = stem.rsplit("-", 1)
    if not base or not variant:
        return None
    return base, variant


def build_complete_groups(image_files: dict[str, Path]) -> tuple[dict[str, GroupFiles], int]:
    grouped: dict[str, dict[str, Path]] = {}

    for stem, path in image_files.items():
        parsed = split_variant(stem)
        if parsed is None:
            continue
        base, variant = parsed
        grouped.setdefault(base, {})[variant] = path

    complete: dict[str, GroupFiles] = {}
    skipped_incomplete = 0
    for base in sorted(grouped.keys()):
        variants = grouped[base]
        if not all(v in variants for v in TARGET_VARIANTS):
            skipped_incomplete += 1
            missing = [v for v in TARGET_VARIANTS if v not in variants]
            print(f"[ALERT] Skipping base '{base}': missing required variants {missing}")
            continue

        complete[base] = GroupFiles(
            all_image=variants["all"],
            seg_image=variants["seg"],
            sunglasses_image=variants["sunglasses"],
        )

    return complete, skipped_incomplete


def mask_to_bbox(mask_path: Path, threshold: int) -> tuple[int, int, int, int] | None:
    with Image.open(mask_path) as mask_img:
        gray = np.array(mask_img.convert("L"), dtype=np.uint8)

    binary = gray > threshold
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return None

    x_low = int(xs.min())
    x_high = int(xs.max())
    y_low = int(ys.min())
    y_high = int(ys.max())

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


def write_data_yaml(output_folder: Path, class_id: int) -> None:
    data_yaml = output_folder / "data.yaml"
    content = f"""path: .
train: images
val: images
names:
  {class_id}: glasses
"""
    data_yaml.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()

    if not args.input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {args.input_folder}")
    if not args.input_folder.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {args.input_folder}")

    out_images = args.output_folder / "images"
    out_labels = args.output_folder / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_files = find_image_files(args.input_folder)
    complete_groups, skipped_incomplete = build_complete_groups(image_files)
    groups = sorted(complete_groups.items(), key=lambda item: item[0])

    if args.test_mode:
        groups = groups[: args.limit]

    print(f"Found {len(image_files)} total images in: {args.input_folder}")
    print(f"Found {len(complete_groups)} complete all/seg/sunglasses groups")
    print(f"Scheduled {len(groups)} groups for processing")
    print(f"Output folder: {args.output_folder}")
    print("Source dataset remains unchanged (read-only access).")

    processed_groups = 0
    written_all = 0
    written_sunglasses = 0
    skipped_empty_seg = 0
    skipped_size_mismatch = 0

    for base, group in groups:
        processed_groups += 1

        with Image.open(group.seg_image) as seg_img:
            seg_w, seg_h = seg_img.size

        with Image.open(group.all_image) as all_img:
            all_w, all_h = all_img.size
        with Image.open(group.sunglasses_image) as sunglasses_img:
            sunglasses_w, sunglasses_h = sunglasses_img.size

        if (all_w, all_h) != (seg_w, seg_h) or (sunglasses_w, sunglasses_h) != (seg_w, seg_h):
            print(
                f"[ALERT] Skipping base '{base}': size mismatch "
                f"seg=({seg_w}x{seg_h}), all=({all_w}x{all_h}), "
                f"sunglasses=({sunglasses_w}x{sunglasses_h})"
            )
            skipped_size_mismatch += 1
            continue

        bbox = mask_to_bbox(group.seg_image, threshold=args.threshold)

        if bbox is None:
            print(f"[ALERT] Skipping base '{base}': seg mask has no foreground")
            skipped_empty_seg += 1
            continue

        bbox = tighten_bbox(
            bbox,
            image_width=seg_w,
            image_height=seg_h,
            tightness=args.tightness,
        )

        label_line = "{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
        for target in (group.all_image, group.sunglasses_image):
            x, y, w, h = bbox_to_yolo(bbox, image_width=seg_w, image_height=seg_h)
            label_text = label_line.format(class_id=args.class_id, x=x, y=y, w=w, h=h)
            label_path = out_labels / f"{target.stem}.txt"
            label_path.write_text(label_text, encoding="utf-8")

            if target is group.all_image:
                written_all += 1
            else:
                written_sunglasses += 1

            if args.copy_images:
                shutil.copy2(target, out_images / target.name)

        if processed_groups % 100 == 0:
            print(
                "Processed "
                f"{processed_groups}/{len(groups)} groups | "
                f"labels(all): {written_all} | "
                f"labels(sunglasses): {written_sunglasses} | "
                f"skipped: {skipped_empty_seg + skipped_size_mismatch}"
            )

    if args.write_data_yaml:
        write_data_yaml(args.output_folder, class_id=args.class_id)

    print("\nDone.")
    print(f"Processed groups: {processed_groups}")
    print(f"Labels written for all: {written_all}")
    print(f"Labels written for sunglasses: {written_sunglasses}")
    print(f"Skipped incomplete groups: {skipped_incomplete}")
    print(f"Skipped empty seg masks: {skipped_empty_seg}")
    print(f"Skipped size mismatches: {skipped_size_mismatch}")
    print(f"Saved labels to: {out_labels}")
    if args.copy_images:
        print(f"Saved images to: {out_images}")
    else:
        print("Image copying disabled (--no-copy-images).")


if __name__ == "__main__":
    main()
