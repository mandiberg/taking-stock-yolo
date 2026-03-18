#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".bmp", ".webp")


def find_image_for_label(label_path: Path) -> Path | None:
    image_dir = Path(str(label_path.parent).replace('/labels', '/images'))
    stem = label_path.stem
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def copy_bucket(rows: list[dict], bucket_name: str, output_root: Path) -> dict:
    bucket_root = output_root / f"suspect_images_{bucket_name}"
    labels_out = bucket_root / "labels"
    images_out = bucket_root / "images"
    labels_out.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)

    labels_copied = 0
    images_copied = 0
    missing_labels = 0
    missing_images = 0

    for row in rows:
        label_path = Path(row["file_path"])
        if not label_path.exists():
            missing_labels += 1
            continue

        shutil.copy2(label_path, labels_out / label_path.name)
        labels_copied += 1

        image_path = find_image_for_label(label_path)
        if image_path is None:
            missing_images += 1
            continue

        shutil.copy2(image_path, images_out / image_path.name)
        images_copied += 1

    return {
        "bucket": bucket_name,
        "labels_folder_exists": labels_out.exists(),
        "images_folder_exists": images_out.exists(),
        "labels_copied": labels_copied,
        "images_copied": images_copied,
        "missing_labels": missing_labels,
        "missing_images": missing_images,
        "labels_out": str(labels_out),
        "images_out": str(images_out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy suspect labels and matching images into high/low confidence buckets.")
    parser.add_argument(
        "--analysis-json",
        default="/Users/michael.mandiberg/Documents/GitHub/taking-stock-yolo/runs/tiny_label_analysis.json",
        help="JSON generated from analyze_tiny_yolo_labels.py.",
    )
    parser.add_argument(
        "--output-root",
        default="/Users/michael.mandiberg/Documents/YOLO_Training_Data/reprocess",
        help="Root folder where suspect_images_high_conf/ and suspect_images_low_conf/ will be created.",
    )
    args = parser.parse_args()

    analysis_json = Path(args.analysis_json)
    output_root = Path(args.output_root)

    if not analysis_json.exists():
        raise FileNotFoundError(f"Analysis JSON not found: {analysis_json}")

    payload = json.loads(analysis_json.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])

    high_conf_rows = [row for row in rows if row.get("high_conf")]
    low_conf_rows = [row for row in rows if row.get("low_conf")]

    high_conf_result = copy_bucket(high_conf_rows, "high_conf", output_root)
    low_conf_result = copy_bucket(low_conf_rows, "low_conf", output_root)

    print(
        {
            "analysis_json": str(analysis_json),
            "output_root": str(output_root),
            "high_conf": high_conf_result,
            "low_conf": low_conf_result,
        }
    )


if __name__ == "__main__":
    main()
