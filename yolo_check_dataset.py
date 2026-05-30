#!/usr/bin/env python3
"""yolo_check_dataset.py

Scan a YOLOv8 dataset and identify corrupt/malformed images and label files.

Checks performed:
- Image file exists for each label and can be opened (PIL verify)
- Label file exists for each image (counts as 'background' if missing)
- Label line format: at least 5 columns (class x y w h)
- Class id is integer (and optionally within range if provided nc)
- bbox values (x,y,w,h) are floats and within a reasonable range (0..1)
- No NaN/inf values

# delete empty label files (optional):
find yolo_dataset/labels -type f -name '*.txt' -empty -delete

Outputs a JSON report `yolo_check_report.json` in the dataset root.
"""

import argparse
import json
import math
from pathlib import Path
from PIL import Image, UnidentifiedImageError


def safe_float(s):
    try:
        v = float(s)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def check_label_contents(label_path, class_map=None):
    """Return list of problems found in a label file (empty list if OK)"""
    problems = []
    try:
        with open(label_path, 'r') as f:
            lines = [ln.rstrip('\n') for ln in f.readlines()]
    except Exception as e:
        return [f'cannot_read_label: {e}']

    if len(lines) == 0:
        return ['empty_label_file']

    for i, line in enumerate(lines, start=1):
        if not line.strip():
            problems.append(f'empty_line_{i}')
            continue
        parts = line.split()
        if len(parts) < 5:
            problems.append(f'bad_columns_line_{i}: expected>=5 got={len(parts)}')
            continue
        # check class id
        cls = parts[0]
        try:
            int_cls = int(cls)
        except Exception:
            problems.append(f'nonint_class_line_{i}: "{cls}"')
            continue

        # check bbox floats
        coords = parts[1:5]
        for j, c in enumerate(coords, start=1):
            v = safe_float(c)
            if v is None:
                problems.append(f'bad_float_line_{i}_col{j+1}: "{c}"')
                continue
            # YOLO relative coords should be between 0 and 1 (some tolerance)
            if not (0.0 <= v <= 1.0):
                problems.append(f'out_of_range_line_{i}_col{j+1}: {v}')

    return problems


def check_image_openable(image_path):
    try:
        with Image.open(image_path) as im:
            im.verify()
        return []
    except UnidentifiedImageError:
        return ['unidentified_image']
    except Exception as e:
        return [f'image_open_error: {e}']


def scan_dataset(root: str, verbose=False):
    root = Path(root)
    report = {
        'root': str(root),
        'splits': {},
    }

    for split in ('train', 'val'):
        images_dir = root / 'images' / split
        labels_dir = root / 'labels' / split
        split_report = {
            'images_count': 0,
            'labels_count': 0,
            'backgrounds': [],
            'missing_images_for_labels': [],
            'corrupt_labels': {},
            'unopenable_images': {},
        }

        if not images_dir.exists() and not labels_dir.exists():
            report['splits'][split] = {'error': 'no_images_or_labels_dir'}
            continue

        # gather images and label filenames
        images = []
        if images_dir.exists():
            for ext in ('*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff','*.webp'):
                images.extend(images_dir.glob(ext))
        images = sorted(images)
        image_stems = {p.stem: str(p) for p in images}
        split_report['images_count'] = len(images)

        labels = []
        if labels_dir.exists():
            labels = sorted(labels_dir.glob('*.txt'))
        split_report['labels_count'] = len(labels)

        # check labels
        for label in labels:
            stem = label.stem
            img_path = image_stems.get(stem)
            if img_path is None:
                split_report['missing_images_for_labels'].append(str(label))
            problems = check_label_contents(str(label))
            if problems:
                split_report['corrupt_labels'][str(label)] = problems

        # check images that have no label (backgrounds)
        for stem, img_path in image_stems.items():
            label_path = labels_dir / (stem + '.txt')
            if not label_path.exists():
                split_report['backgrounds'].append(img_path)

        # check if images open fine
        for stem, img_path in image_stems.items():
            probs = check_image_openable(img_path)
            if probs:
                split_report['unopenable_images'][img_path] = probs

        report['splits'][split] = split_report

    total_images = sum(report['splits'].get(s, {}).get('images_count', 0) for s in report['splits'])
    total_labels = sum(report['splits'].get(s, {}).get('labels_count', 0) for s in report['splits'])
    report['summary'] = {'total_images': total_images, 'total_labels': total_labels}
    return report


def find_associated_images(images_dir: Path, stem: str) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(path for path in images_dir.glob(f'{stem}.*') if path.is_file())


def delete_corrupt_labels_and_images(root: Path, report: dict) -> dict[str, int]:
    deleted_labels = 0
    deleted_images = 0

    for split, info in report['splits'].items():
        if 'error' in info:
            continue

        images_dir = root / 'images' / split
        corrupt_label_paths = sorted(info['corrupt_labels'].keys())

        for label_path_str in corrupt_label_paths:
            label_path = Path(label_path_str)
            if label_path.exists():
                label_path.unlink()
                deleted_labels += 1
                print(f"Deleted corrupt label: {label_path}")

            for image_path in find_associated_images(images_dir, label_path.stem):
                if image_path.exists():
                    image_path.unlink()
                    deleted_images += 1
                    print(f"Deleted associated image: {image_path}")

    return {'deleted_labels': deleted_labels, 'deleted_images': deleted_images}


def main():
    parser = argparse.ArgumentParser(description='Check YOLO dataset images/labels for corruption and format issues.')
    parser.add_argument('dataset_root', nargs='?', default='yolo_dataset', help='Path to YOLO dataset root (default: yolo_dataset)')
    parser.add_argument('--out', '-o', help='Path to write JSON report (default: <dataset_root>/yolo_check_report.json)')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument(
        '--delete-corrupt',
        action='store_true',
        help='Delete corrupt label files and any same-stem image files in the matching split.',
    )
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    if not root.exists():
        print(f"Dataset root does not exist: {root}")
        raise SystemExit(1)

    report = scan_dataset(str(root), verbose=args.verbose)

    out_path = Path(args.out) if args.out else root / 'yolo_check_report.json'
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Wrote report: {out_path}")
    for split, info in report['splits'].items():
        if 'error' in info:
            print(f"{split}: {info['error']}")
            continue
        imgs = info['images_count']
        labs = info['labels_count']
        backgrounds = len(info['backgrounds'])
        corrupt_labels = len(info['corrupt_labels'])
        unopenable = len(info['unopenable_images'])
        missing_images_for_labels = len(info['missing_images_for_labels'])
        print(f"{split}: images={imgs}, labels={labs}, backgrounds={backgrounds}, corrupt_labels={corrupt_labels}, unopenable_images={unopenable}, missing_images_for_labels={missing_images_for_labels}")
        print("corrupt_labels:")
        for label, problems in info['corrupt_labels'].items():
            print(f"  {label}: {', '.join(problems)}")

    if args.delete_corrupt:
        deletion_summary = delete_corrupt_labels_and_images(root, report)
        print(
            'Deletion summary: '
            f"labels={deletion_summary['deleted_labels']}, images={deletion_summary['deleted_images']}"
        )

if __name__ == '__main__':
    main()
