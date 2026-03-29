import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_SOURCE = Path("/Users/michaelmandiberg/Documents/YOLO_Training_Data/sorted_images_flowers/none_bag_groceries_flowers")


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def read_nonempty_lines(label_path: Path) -> list[str]:
    with open(label_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def parse_class_rows(lines: list[str], label_path: Path) -> dict[str, list[str]]:
    rows_by_class = defaultdict(list)
    for idx, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) < 5:
            print(f"Skipping malformed label row {label_path}:{idx}: {line}")
            continue
        class_id = parts[0]
        rows_by_class[class_id].append(line)
    return rows_by_class


def ensure_class_dirs(output_root: Path, class_id: str) -> tuple[Path, Path]:
    class_root = output_root / f"class_{class_id}"
    class_images = class_root / "images"
    class_labels = class_root / "labels"
    class_images.mkdir(parents=True, exist_ok=True)
    class_labels.mkdir(parents=True, exist_ok=True)
    return class_images, class_labels


def copy_to_decoys(image_path: Path, label_path: Path | None, output_root: Path) -> None:
    decoys_images = output_root / "decoys" / "images"
    decoys_labels = output_root / "decoys" / "labels"
    decoys_images.mkdir(parents=True, exist_ok=True)
    decoys_labels.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, decoys_images / image_path.name)
    if label_path and label_path.exists():
        shutil.copy2(label_path, decoys_labels / label_path.name)


def split_folders_by_class(source_root: Path, output_root: Path) -> None:
    images_dir = source_root / "images"
    labels_dir = source_root / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    image_files = sorted([p for p in images_dir.iterdir() if p.is_file() and is_image_file(p)])

    stats = Counter()
    class_file_counts = Counter()

    for image_path in image_files:
        stats["images_seen"] += 1
        label_path = labels_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            copy_to_decoys(image_path, None, output_root)
            stats["decoys_missing_label"] += 1
            print(f"Decoy (missing label): {image_path.name}")
            continue

        lines = read_nonempty_lines(label_path)
        if not lines:
            copy_to_decoys(image_path, label_path, output_root)
            stats["decoys_empty_label"] += 1
            print(f"Decoy (empty label): {image_path.name}")
            continue

        rows_by_class = parse_class_rows(lines, label_path)
        if not rows_by_class:
            copy_to_decoys(image_path, label_path, output_root)
            stats["decoys_malformed_label"] += 1
            print(f"Decoy (malformed-only label): {image_path.name}")
            continue

        for class_id, class_rows in rows_by_class.items():
            class_images_dir, class_labels_dir = ensure_class_dirs(output_root, class_id)
            shutil.copy2(image_path, class_images_dir / image_path.name)
            out_label_path = class_labels_dir / f"{image_path.stem}.txt"
            with open(out_label_path, "w") as f:
                f.write("\n".join(class_rows) + "\n")

            class_file_counts[class_id] += 1
            stats["class_assignments"] += 1

    print("\nSplit complete")
    print(f"Source: {source_root}")
    print(f"Output: {output_root}")
    print(f"Images scanned: {stats['images_seen']}")
    print(f"Class assignments written: {stats['class_assignments']}")
    print(f"Decoys (missing label): {stats['decoys_missing_label']}")
    print(f"Decoys (empty label): {stats['decoys_empty_label']}")
    print(f"Decoys (malformed-only label): {stats['decoys_malformed_label']}")

    if class_file_counts:
        print("\nPer-class file counts:")
        for class_id in sorted(class_file_counts, key=lambda x: int(x) if x.isdigit() else x):
            print(f"  class_{class_id}: {class_file_counts[class_id]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a mixed YOLO images/labels folder into per-class folders plus decoys."
    )
    parser.add_argument(
        "source_root",
        nargs="?",
        default=str(DEFAULT_SOURCE),
        help="Folder containing 'images/' and 'labels/'",
    )
    parser.add_argument(
        "--output-root",
        "-o",
        default=None,
        help="Output folder. Default: <source_root>/split_by_class",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser().resolve()
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else source_root / "split_by_class"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    split_folders_by_class(source_root, output_root)


if __name__ == "__main__":
    main()
