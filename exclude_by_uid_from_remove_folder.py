#!/usr/bin/env python3
"""
Move files from YOLO training datasets into an excluded folder using UIDs
extracted from filenames found in a remove folder.

Default behavior is testing mode (dry run): prints planned moves, does not move files.

Usage examples:
python exclude_by_uid_from_remove_folder.py
python exclude_by_uid_from_remove_folder.py --execute
python exclude_by_uid_from_remove_folder.py --testing --stop-after 50
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

REMOVE_FOLDER = Path("/Users/michaelmandiberg/Documents/images_to_remove")
YOLO_TRAINING_DATA = Path("/Users/michaelmandiberg/Documents/YOLO_Training_Data")
EXCLUDE_FOLDER = Path("/Users/michaelmandiberg/Documents/images_excluded")
STOP_AFTER = 20

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
LABEL_EXTS = {".txt"}
MIN_UID_DIGITS = 5


def is_hidden_or_macos_artifact(path: Path) -> bool:
    name = path.name
    if name.startswith("."):
        return True
    if name.startswith("._"):
        return True
    if any(part.startswith(".") for part in path.parts):
        return True
    return False


def extract_image_id(filename: str) -> Optional[str]:
    name = Path(filename).name

    m = re.search(r"_(\d+)_", name)
    if m:
        candidate = m.group(1)
        if len(candidate) >= MIN_UID_DIGITS:
            return candidate

    m = re.match(r"^(\d+)(?:_|$)", name)
    if m:
        candidate = m.group(1)
        if len(candidate) >= MIN_UID_DIGITS:
            return candidate

    seqs = re.findall(r"(\d+)", name)
    if seqs:
        candidate = max(seqs, key=len)
        if len(candidate) >= MIN_UID_DIGITS:
            return candidate

    return None


def build_original_filepaths(root: Path) -> List[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and not is_hidden_or_macos_artifact(path)
    ]


def build_remove_filenames(remove_folder: Path) -> List[str]:
    return [
        path.name
        for path in sorted(remove_folder.iterdir())
        if path.is_file() and not is_hidden_or_macos_artifact(path)
    ]


def classify_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTS:
        return "image"
    if suffix in LABEL_EXTS:
        return "label"
    return "other"


def build_uid_index(original_filepaths: List[Path]) -> Dict[str, Dict[str, List[Path]]]:
    uid_index: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: {"images": [], "labels": []})

    for path in original_filepaths:
        file_type = classify_file(path)
        if file_type == "other":
            continue

        uid = extract_image_id(path.name)
        if not uid:
            continue

        if file_type == "image":
            uid_index[uid]["images"].append(path)
        else:
            uid_index[uid]["labels"].append(path)

    return uid_index


def unique_destination_path(destination_folder: Path, source_path: Path) -> Path:
    destination = destination_folder / source_path.name
    if not destination.exists():
        return destination

    stem = source_path.stem
    suffix = source_path.suffix
    counter = 1
    while True:
        candidate = destination_folder / f"{stem}__dup{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def move_or_print(source_path: Path, destination_folder: Path, testing: bool) -> Path:
    destination_folder.mkdir(parents=True, exist_ok=True)
    destination_path = unique_destination_path(destination_folder, source_path)

    if testing:
        print(f"[TEST] move {source_path} -> {destination_path}")
    else:
        source_path.rename(destination_path)
        print(f"Moved {source_path} -> {destination_path}")

    return destination_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move images/labels from YOLO_TRAINING_DATA to EXCLUDE_FOLDER based on UIDs in REMOVE_FOLDER filenames."
    )
    parser.add_argument("--remove-folder", type=Path, default=REMOVE_FOLDER)
    parser.add_argument("--source-root", type=Path, default=YOLO_TRAINING_DATA)
    parser.add_argument("--exclude-folder", type=Path, default=EXCLUDE_FOLDER)
    parser.add_argument("--testing", action="store_true", help="Testing mode: print paths, do not move files.")
    parser.add_argument("--execute", action="store_true", help="Execute moves (overrides testing mode).")
    parser.add_argument("--stop-after", type=int, default=STOP_AFTER, help="In testing mode, stop after this many remove files.")
    args = parser.parse_args()

    testing = True
    if args.execute:
        testing = False
    elif args.testing:
        testing = True

    remove_folder = args.remove_folder.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    exclude_folder = args.exclude_folder.expanduser().resolve()
    exclude_images = exclude_folder / "images"
    exclude_labels = exclude_folder / "labels"
    exclude_decoys = exclude_folder / "decoys"

    if not remove_folder.exists() or not remove_folder.is_dir():
        raise FileNotFoundError(f"Remove folder not found or not a directory: {remove_folder}")
    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f"Source root not found or not a directory: {source_root}")

    original_filepaths = build_original_filepaths(source_root)
    remove_filenames = build_remove_filenames(remove_folder)

    uid_index = build_uid_index(original_filepaths)

    print(f"Scanned source files: {len(original_filepaths)}")
    print(f"Remove filenames found: {len(remove_filenames)}")
    print(f"UIDs indexed in source: {len(uid_index)}")

    processed_remove_files = 0
    matched_uids = 0
    moved_images = 0
    moved_labels = 0
    moved_decoys = 0
    missing_uids = 0
    skipped_orphan_labels = 0

    moved_sources: Set[Path] = set()

    for remove_name in remove_filenames:
        uid = extract_image_id(remove_name)
        processed_remove_files += 1

        if not uid:
            print(f"No UID found in remove filename: {remove_name}")
            continue

        if uid not in uid_index:
            print(f"UID not found in source index: {uid} (from {remove_name})")
            missing_uids += 1
            if testing and processed_remove_files >= args.stop_after:
                print(f"Testing stop reached at {processed_remove_files} remove files.")
                break
            continue

        matches = uid_index[uid]
        images = matches["images"]
        labels = matches["labels"]
        matched_uids += 1

        print(f"UID {uid} from {remove_name}: {len(images)} image(s), {len(labels)} label(s)")

        labels_by_stem: Dict[str, List[Path]] = defaultdict(list)
        for label_path in labels:
            if label_path in moved_sources:
                continue
            labels_by_stem[label_path.stem].append(label_path)

        for image_path in images:
            if image_path in moved_sources:
                continue

            paired_label: Optional[Path] = None
            candidates = labels_by_stem.get(image_path.stem, [])
            while candidates:
                candidate = candidates.pop(0)
                if candidate not in moved_sources:
                    paired_label = candidate
                    break

            if paired_label is not None:
                move_or_print(image_path, exclude_images, testing)
                move_or_print(paired_label, exclude_labels, testing)
                moved_sources.add(image_path)
                moved_sources.add(paired_label)
                moved_images += 1
                moved_labels += 1
            else:
                move_or_print(image_path, exclude_decoys, testing)
                moved_sources.add(image_path)
                moved_decoys += 1

        for label_path in labels:
            if label_path in moved_sources:
                continue
            skipped_orphan_labels += 1
            print(f"Unpaired label not moved (no matching image stem): {label_path}")

        if testing and processed_remove_files >= args.stop_after:
            print(f"Testing stop reached at {processed_remove_files} remove files.")
            break

    print("\nDone")
    print(f"Mode: {'testing' if testing else 'execute'}")
    print(f"Processed remove filenames: {processed_remove_files}")
    print(f"UID matches found: {matched_uids}")
    print(f"UIDs missing in source: {missing_uids}")
    print(f"Image files {'planned' if testing else 'moved'}: {moved_images}")
    print(f"Label files {'planned' if testing else 'moved'}: {moved_labels}")
    print(f"Decoy image files {'planned' if testing else 'moved'}: {moved_decoys}")
    print(f"Unpaired labels skipped: {skipped_orphan_labels}")
    print(f"Exclude images folder: {exclude_images}")
    print(f"Exclude labels folder: {exclude_labels}")
    print(f"Exclude decoys folder: {exclude_decoys}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
