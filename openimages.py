import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import fiftyone.zoo as foz


# Toggle this for small, fast local pulls.
TESTING_MODE = True
LIMIT = 200
DEFAULT_TARGET_CLASSES = ["Tablet computer", "Calculator"]
DEFAULT_SPLITS = ["train", "validation"]
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = "yolo_dataset2/openimages_raw"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Open Images detections and export a single-folder YOLO dataset"
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=",".join(DEFAULT_TARGET_CLASSES),
        help="Comma-separated Open Images class names",
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
    parser.set_defaults(testing_mode=TESTING_MODE)
    return parser.parse_args()


def resolve_target_classes(requested_classes):
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

    if missing:
        raise ValueError(
            "Open Images classes not found: "
            f"{missing}. Check exact class names in Open Images taxonomy."
        )

    return resolved


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


def select_balanced(candidates, target_classes, limit, seed):
    random.seed(seed)
    shuffled = list(candidates)
    random.shuffle(shuffled)

    class_to_indices = defaultdict(list)
    for idx, c in enumerate(shuffled):
        for label in c["classes_present"]:
            class_to_indices[label].append(idx)

    selected = set()
    counts = {cls: 0 for cls in target_classes}

    while True:
        deficits = {cls: max(0, limit - counts[cls]) for cls in target_classes}
        if all(v == 0 for v in deficits.values()):
            break

        needed = sorted(target_classes, key=lambda c: deficits[c], reverse=True)
        picked = False

        for cls in needed:
            if deficits[cls] == 0:
                continue

            best_idx = None
            best_score = -1
            for idx in class_to_indices.get(cls, []):
                if idx in selected:
                    continue
                score = sum(deficits.get(label, 0) for label in shuffled[idx]["classes_present"])
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                selected.add(best_idx)
                for label in shuffled[best_idx]["classes_present"]:
                    if label in counts:
                        counts[label] += 1
                picked = True
                break

        if not picked:
            break

    return [shuffled[i] for i in sorted(selected)], counts


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
    target_classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    source_splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    if not target_classes:
        raise ValueError("No classes specified. Use --classes.")
    if not source_splits:
        raise ValueError("No source splits specified. Use --splits.")

    resolved_classes = resolve_target_classes(target_classes)

    max_samples_per_split = args.max_samples_per_split
    if args.testing_mode and max_samples_per_split is None:
        desired = args.limit * max(1, len(resolved_classes))
        max_samples_per_split = max(500, desired * 2)

    print("=== Open Images Download Config ===")
    print(f"Testing mode: {args.testing_mode}")
    print(f"Per-class limit: {args.limit if args.testing_mode else 'disabled'}")
    print(f"Target classes: {resolved_classes}")
    print(f"Source splits: {source_splits}")
    print(f"max_samples_per_split: {max_samples_per_split}")
    print(f"Output dir: {args.output_dir}")

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

    if args.testing_mode:
        selected, counts = select_balanced(
            candidates=candidates,
            target_classes=resolved_classes,
            limit=args.limit,
            seed=args.seed,
        )
    else:
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