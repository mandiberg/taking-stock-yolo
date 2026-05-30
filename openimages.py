import argparse
import random
import shutil
import time
from pathlib import Path

import fiftyone.zoo as foz

from class_map_utils import load_class_records


# Toggle this for small, fast local pulls.
TESTING_MODE = True
LIMIT = 1000

DEFAULT_SPLITS = ["train", "validation"]
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = Path.home() / "Documents/YOLO_Training_Data/openimages_raw"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Open Images detections and export a single-folder YOLO dataset"
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="",
        help=(
            "Comma-separated Open Images class names. "
            "If omitted, classes are sourced from custom_class_map.json"
        ),
    )
    parser.add_argument(
        "--use-custom-map",
        dest="use_custom_map",
        action="store_true",
        help="Use canonical and alias names from custom_class_map.json when --classes is omitted",
    )
    parser.add_argument(
        "--no-use-custom-map",
        dest="use_custom_map",
        action="store_false",
        help="Require explicit --classes and do not auto-load class names from custom map",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder that will contain images/, labels/, and classes.txt",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default=",".join(DEFAULT_SPLITS),
        help="Comma-separated source splits to load from Open Images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible selection",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=LIMIT,
        help="Per-class image cap used when testing mode is enabled",
    )
    parser.add_argument(
        "--testing-mode",
        dest="testing_mode",
        action="store_true",
        help="Enable per-class cap (default from TESTING_MODE constant)",
    )
    parser.add_argument(
        "--production-mode",
        dest="testing_mode",
        action="store_false",
        help="Disable per-class cap and keep all matching samples",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Optional Open Images max_samples for each split load",
    )
    parser.set_defaults(testing_mode=TESTING_MODE, use_custom_map=True)
    return parser.parse_args()


def classes_from_custom_map():
    names = []
    for row in load_class_records():
        canonical = str(row.get("name", "")).strip()
        if canonical:
            names.append(canonical)

        aliases = row.get("aliases", [])
        if isinstance(aliases, list):
            for alias in aliases:
                alias_text = str(alias).strip()
                if alias_text:
                    names.append(alias_text)

    ordered_unique = []
    seen = set()
    for name in names:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered_unique.append(name)

    return ordered_unique


def resolve_target_classes(requested_classes, strict=True):
    probe = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections"],
        max_samples=1,
    )
    available = probe.default_classes
    name_map = {c.lower(): c for c in available}

    resolved = []
    missing = []
    for cls in requested_classes:
        key = cls.strip().lower()
        if key in name_map:
            resolved.append(name_map[key])
        else:
            missing.append(cls)

    if missing and strict:
        raise ValueError(
            "Open Images classes not found: "
            f"{missing}. Check exact class names in Open Images taxonomy."
        )

    return resolved, missing


def print_class_resolution_report(requested_classes, resolved_classes, missing_classes):
    print("=== Class Resolution Report ===")
    print(f"Requested names: {len(requested_classes)}")
    print(f"Matched Open Images names: {len(resolved_classes)}")
    print(f"Skipped (not in Open Images): {len(missing_classes)}")

    if missing_classes:
        preview_limit = 100
        preview = ", ".join(missing_classes[:preview_limit])
        if len(missing_classes) > preview_limit:
            preview += ", ..."
        print(f"Skipped preview: {preview}")


def build_candidates(datasets, target_classes):
    by_path = {}

    for split_name, dataset in datasets:
        for sample in dataset:
            detections = sample.ground_truth.detections if sample.ground_truth else []
            target_detections = [d for d in detections if d.label in target_classes]
            if not target_detections:
                continue

            classes_present = sorted({d.label for d in target_detections})
            image_id = getattr(sample, "open_images_id", None)
            if image_id is None:
                image_id = getattr(sample, "id", None)
            if image_id is None:
                image_id = Path(sample.filepath).stem

            info = {
                "filepath": sample.filepath,
                "split": split_name,
                "classes_present": classes_present,
                "detections": target_detections,
                "image_id": image_id,
            }
            by_path[sample.filepath] = info

    return list(by_path.values())


def detection_key(det):
    x, y, w, h = det.bounding_box
    return det.label, round(x, 6), round(y, 6), round(w, 6), round(h, 6)


def merge_sample_record(existing, incoming):
    existing["classes_present"] = sorted(set(existing["classes_present"]) | set(incoming["classes_present"]))

    seen = {detection_key(d) for d in existing["detections"]}
    for det in incoming["detections"]:
        key = detection_key(det)
        if key in seen:
            continue
        existing["detections"].append(det)
        seen.add(key)


def write_yolo_dataset(selected_samples, output_dir, target_classes):
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    class_to_id = {cls: idx for idx, cls in enumerate(target_classes)}

    for sample in selected_samples:
        src = Path(sample["filepath"])
        ext = src.suffix.lower() if src.suffix else ".jpg"
        stem = f"{sample['split']}_{sample['image_id']}"
        safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)

        dst_image = images_dir / f"{safe_stem}{ext}"
        if not dst_image.exists():
            shutil.copy2(src, dst_image)

        label_lines = []
        for det in sample["detections"]:
            if det.label not in class_to_id:
                continue
            cls_id = class_to_id[det.label]
            x, y, w, h = det.bounding_box
            x_center = x + (w / 2.0)
            y_center = y + (h / 2.0)
            label_lines.append(
                f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            )

        label_path = labels_dir / f"{safe_stem}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))

    classes_path = output_dir / "classes.txt"
    with open(classes_path, "w", encoding="utf-8") as f:
        f.write("\n".join(target_classes) + "\n")


def main():
    args = parse_args()
    explicit_classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    source_splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    if not source_splits:
        raise ValueError("No source splits specified. Use --splits.")

    used_custom_map = False
    if explicit_classes:
        target_classes = explicit_classes
    elif args.use_custom_map:
        target_classes = classes_from_custom_map()
        used_custom_map = True
    else:
        raise ValueError("No classes specified. Use --classes or enable --use-custom-map.")

    if not target_classes:
        raise ValueError("No target classes resolved from inputs.")

    resolved_classes, missing_classes = resolve_target_classes(
        target_classes,
        strict=not used_custom_map,
    )

    print_class_resolution_report(
        requested_classes=target_classes,
        resolved_classes=resolved_classes,
        missing_classes=missing_classes,
    )

    if used_custom_map and missing_classes:
        print("Note: skipped names came from custom_class_map.json")
    if not resolved_classes:
        raise RuntimeError("No target classes matched Open Images taxonomy after filtering.")

    max_samples_per_split = args.max_samples_per_split
    if args.testing_mode and max_samples_per_split is None:
        # Pull a larger candidate pool per class, then sample per-class downstream.
        max_samples_per_split = max(1000, args.limit * 2)

    print("=== Open Images Download Config ===")
    print(f"Testing mode: {args.testing_mode}")
    print(f"Per-class limit: {args.limit if args.testing_mode else 'disabled'}")
    print(f"Class source: {'custom_class_map.json' if used_custom_map else '--classes'}")
    print(f"Target classes: {resolved_classes}")
    print(f"Source splits: {source_splits}")
    print(f"max_samples_per_split: {max_samples_per_split}")
    print(f"Output dir: {args.output_dir}")

    # Prevent FiftyOne from reusing stale per-split datasets across different classes.
    run_tag = str(int(time.time()))

    if args.testing_mode:
        # In testing mode, sample each class independently up to LIMIT.
        selected_by_path = {}
        counts = {cls: 0 for cls in resolved_classes}

        for cls_index, cls_name in enumerate(resolved_classes):
            print(f"\nSampling class '{cls_name}'...")
            class_datasets = []
            safe_cls = "".join(ch if ch.isalnum() else "_" for ch in cls_name.lower())
            for split in source_splits:
                print(f"Loading split '{split}' for class '{cls_name}'...")
                dataset_name = (
                    f"open-images-v7-{split}-{safe_cls}-"
                    f"{max_samples_per_split}-{args.seed + cls_index}-{run_tag}"
                )
                ds = foz.load_zoo_dataset(
                    "open-images-v7",
                    split=split,
                    label_types=["detections"],
                    classes=[cls_name],
                    only_matching=True,
                    shuffle=True,
                    seed=args.seed + cls_index,
                    max_samples=max_samples_per_split,
                    dataset_name=dataset_name,
                )
                class_datasets.append((split, ds))
                print(f"Loaded {len(ds)} samples from {split} for class '{cls_name}'")

            # Keep all target-class detections for selected images so downstream routing
            # can place the same image into all relevant class folders.
            class_candidates = build_candidates(class_datasets, resolved_classes)
            class_candidates = [
                sample for sample in class_candidates if cls_name in sample["classes_present"]
            ]
            random.Random(args.seed + cls_index).shuffle(class_candidates)
            take_n = min(args.limit, len(class_candidates))
            class_selected = class_candidates[:take_n]
            counts[cls_name] = len(class_selected)
            print(f"Selected {take_n} images for class '{cls_name}'")

            for sample in class_selected:
                key = sample["filepath"]
                if key in selected_by_path:
                    merge_sample_record(selected_by_path[key], sample)
                else:
                    selected_by_path[key] = {
                        "filepath": sample["filepath"],
                        "split": sample["split"],
                        "classes_present": list(sample["classes_present"]),
                        "detections": list(sample["detections"]),
                        "image_id": sample["image_id"],
                    }

        selected = list(selected_by_path.values())
        if not selected:
            raise RuntimeError("No samples were selected for any class.")
    else:
        datasets = []
        for split in source_splits:
            print(f"Loading split '{split}'...")
            ds = foz.load_zoo_dataset(
                "open-images-v7",
                split=split,
                label_types=["detections"],
                classes=resolved_classes,
                only_matching=True,
                shuffle=True,
                seed=args.seed,
                max_samples=max_samples_per_split,
            )
            datasets.append((split, ds))
            print(f"Loaded {len(ds)} samples from {split}")

        candidates = build_candidates(datasets, resolved_classes)
        print(f"Candidate samples containing target classes: {len(candidates)}")

        if not candidates:
            raise RuntimeError("No candidates found for target classes.")

        selected = candidates
        counts = {cls: 0 for cls in resolved_classes}
        for sample in selected:
            for cls in sample["classes_present"]:
                if cls in counts:
                    counts[cls] += 1

    write_yolo_dataset(selected, args.output_dir, resolved_classes)

    print("=== Export Summary ===")
    print(f"Exported images: {len(selected)}")
    print(f"Images folder: {Path(args.output_dir) / 'images'}")
    print(f"Labels folder: {Path(args.output_dir) / 'labels'}")
    print(f"classes.txt: {Path(args.output_dir) / 'classes.txt'}")
    print("Per-class selected image counts:")
    for cls in resolved_classes:
        print(f"  - {cls}: {counts.get(cls, 0)}")


if __name__ == "__main__":
    main()