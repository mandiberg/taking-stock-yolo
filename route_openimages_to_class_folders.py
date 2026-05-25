#!/usr/bin/env python3
"""Route staged Open Images YOLO labels into per-class folders.

Expected staging layout:
  <staging_dir>/
    images/
    labels/
    classes.txt

This script maps local class indices from classes.txt to your global schema class IDs,
then writes per-class datasets:
  <output_root>/<class_folder>/images/*.jpg
  <output_root>/<class_folder>/labels/*.txt

By default it runs as a dry run. Use --apply to write files.


"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from class_map_utils import get_id_to_name, resolve_class_name_to_id


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class TargetClass:
    source_name: str
    global_id: int
    folder_name: str
    canonical_name: str


def extract_uid_candidates(value: str) -> list[str]:
    """Extract likely UID tokens from a CSV cell or filename-like value."""
    text = value.strip().strip('"').strip("'").lower()
    if not text:
        return []

    base = Path(text).name
    stem = Path(base).stem

    # Prefer long hex substrings like Open Images IDs.
    hex_hits = re.findall(r"[0-9a-f]{12,}", stem)
    if hex_hits:
        return sorted(set(hex_hits))

    # Fallback: long alphanumeric tokens if IDs are not strictly hex.
    token_hits = re.findall(r"[a-z0-9]{12,}", stem)
    return sorted(set(token_hits))


def load_excluded_uids(exclude_csv: Path | None) -> set[str]:
    if exclude_csv is None:
        return set()
    if not exclude_csv.exists():
        raise FileNotFoundError(f"exclude.csv not found: {exclude_csv}")

    uids: set[str] = set()
    with exclude_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            for cell in row:
                for uid in extract_uid_candidates(cell):
                    uids.add(uid)

    return uids


def stem_contains_excluded_uid(stem: str, excluded_uids: set[str]) -> bool:
    if not excluded_uids:
        return False
    s = stem.lower()
    return any(uid in s for uid in excluded_uids)


def make_folder_name(class_id: int, canonical_name: str) -> str:
    return f"{class_id}_{canonical_name}"


def build_targets_from_source_classes(source_classes: list[str]) -> list[TargetClass]:
    id_to_name = get_id_to_name()
    targets = []

    for source_name in source_classes:
        source_name = source_name.strip()
        if not source_name:
            continue

        resolved_id = resolve_class_name_to_id(source_name)
        if resolved_id is None:
            raise ValueError(
                f"Could not resolve source class '{source_name}' in custom_class_map.json"
            )

        canonical_name = id_to_name.get(resolved_id, f"class_{resolved_id}")
        folder_name = make_folder_name(resolved_id, canonical_name)

        targets.append(
            TargetClass(
                source_name=source_name,
                global_id=resolved_id,
                folder_name=folder_name,
                canonical_name=canonical_name,
            )
        )

    if not targets:
        raise ValueError("No source classes provided. Use --source-class at least once.")

    return targets


def build_targets_from_staging_classes(staging_classes: list[str]) -> tuple[list[TargetClass], list[str]]:
    """
    Resolve all staging classes against canonical map.
    Returns (targets, unresolved_source_names).
    """
    resolved_targets: list[TargetClass] = []
    unresolved: list[str] = []

    for source_name in staging_classes:
        try:
            target = build_targets_from_source_classes([source_name])[0]
            resolved_targets.append(target)
        except ValueError:
            unresolved.append(source_name)

    return resolved_targets, unresolved


def load_staging_classes(classes_file: Path) -> list[str]:
    if not classes_file.exists():
        raise FileNotFoundError(f"classes.txt not found: {classes_file}")

    return [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_label_file(path: Path) -> list[tuple[int, float, float, float, float]]:
    records = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        s = raw.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_no} expected 5 columns, found {len(parts)}")

        try:
            local_id = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{path}:{line_no} has invalid numeric values: {raw}") from exc

        records.append((local_id, x, y, w, h))

    return records


def find_image_for_stem(images_dir: Path, stem: str) -> Path | None:
    for ext in sorted(IMAGE_EXTS):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    for candidate in images_dir.glob(f"{stem}.*"):
        if candidate.suffix.lower() in IMAGE_EXTS:
            return candidate

    return None


def merge_label_lines(existing_path: Path, new_lines: list[str], overwrite: bool) -> tuple[list[str], bool]:
    """
    Returns (final_lines, changed).
    If overwrite=True, final_lines is exactly new_lines.
    If overwrite=False and file exists, merges + deduplicates lines.
    """
    if overwrite or not existing_path.exists():
        return new_lines, True

    existing_lines = [
        line.strip()
        for line in existing_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    merged = list(existing_lines)
    seen = set(existing_lines)
    changed = False
    for line in new_lines:
        if line in seen:
            continue
        merged.append(line)
        seen.add(line)
        changed = True

    return merged, changed


def route(
    staging_dir: Path,
    output_root: Path,
    mappings: list[TargetClass],
    apply: bool,
    overwrite: bool,
    excluded_uids: set[str] | None = None,
) -> None:
    images_dir = staging_dir / "images"
    labels_dir = staging_dir / "labels"
    classes_file = staging_dir / "classes.txt"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Expected images/ and labels/ under {staging_dir}")

    staging_classes = load_staging_classes(classes_file)

    source_name_to_local = {name: idx for idx, name in enumerate(staging_classes)}
    local_to_target: dict[int, TargetClass] = {}

    missing_sources = []
    for target in mappings:
        if target.source_name not in source_name_to_local:
            missing_sources.append(target.source_name)
            continue
        local_to_target[source_name_to_local[target.source_name]] = target

    if missing_sources:
        raise ValueError(
            "Mapping source classes not found in staging classes.txt: "
            f"{missing_sources}. Found: {staging_classes}"
        )

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No label files found in {labels_dir}")

    files_scanned = 0
    images_missing = 0
    malformed_files = 0

    class_image_counts: Counter[int] = Counter()
    class_box_counts: Counter[int] = Counter()
    unmapped_local_ids: Counter[int] = Counter()

    write_ops = 0
    skip_existing = 0
    merged_label_files = 0
    excluded_files = 0

    excluded_uids = excluded_uids or set()

    for label_path in label_files:
        files_scanned += 1
        stem = label_path.stem
        if stem_contains_excluded_uid(stem, excluded_uids):
            excluded_files += 1
            continue

        image_path = find_image_for_stem(images_dir, stem)
        if image_path is None:
            images_missing += 1
            continue

        try:
            records = read_label_file(label_path)
        except ValueError:
            malformed_files += 1
            continue

        grouped: dict[TargetClass, list[tuple[float, float, float, float]]] = defaultdict(list)
        for local_id, x, y, w, h in records:
            target = local_to_target.get(local_id)
            if target is None:
                unmapped_local_ids[local_id] += 1
                continue
            grouped[target].append((x, y, w, h))

        if not grouped:
            continue

        # Build one complete mapped label set for this image.
        all_lines: list[str] = []
        for target, boxes in grouped.items():
            class_box_counts[target.global_id] += len(boxes)
            for x, y, w, h in boxes:
                all_lines.append(f"{target.global_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        # Deduplicate lines so repeated mappings do not produce duplicate boxes.
        all_lines = sorted(set(all_lines))

        # Route this image into each triggered target folder, but write full labels each time.
        routed_targets: dict[int, TargetClass] = {}
        for target in grouped:
            routed_targets[target.global_id] = target

        for global_id, target in sorted(routed_targets.items()):
            class_image_counts[global_id] += 1

            out_dir = output_root / target.folder_name
            out_images = out_dir / "images"
            out_labels = out_dir / "labels"

            out_image_path = out_images / image_path.name
            out_label_path = out_labels / f"{stem}.txt"

            if apply:
                out_images.mkdir(parents=True, exist_ok=True)
                out_labels.mkdir(parents=True, exist_ok=True)

                if overwrite or not out_image_path.exists():
                    shutil.copy2(image_path, out_image_path)

                final_lines, changed = merge_label_lines(
                    existing_path=out_label_path,
                    new_lines=all_lines,
                    overwrite=overwrite,
                )

                if out_label_path.exists() and not overwrite and changed:
                    merged_label_files += 1

                if overwrite or changed or not out_label_path.exists():
                    out_label_path.write_text("\n".join(final_lines) + "\n", encoding="utf-8")
                    write_ops += 1
                else:
                    skip_existing += 1

    print("=== Open Images Route Summary ===")
    print(f"Mode: {'APPLY' if apply else 'DRY RUN'}")
    print(f"Staging dir: {staging_dir}")
    print(f"Output root: {output_root}")
    print(f"Files scanned: {files_scanned}")
    print(f"Missing images for labels: {images_missing}")
    print(f"Malformed label files: {malformed_files}")
    print(f"Excluded by UID: {excluded_files}")
    print(f"Write operations: {write_ops}")
    print(f"Skipped existing outputs: {skip_existing}")
    print(f"Merged existing label files: {merged_label_files}")

    print("Local class map from classes.txt:")
    for idx, name in enumerate(staging_classes):
        print(f"  {idx}: {name}")

    print("Applied target mapping:")
    for local_id, target in sorted(local_to_target.items()):
        print(
            f"  local {local_id} ({target.source_name}) "
            f"-> global {target.global_id} ({target.folder_name})"
        )

    if unmapped_local_ids:
        print(f"Unmapped local IDs encountered: {dict(sorted(unmapped_local_ids.items()))}")
    else:
        print("Unmapped local IDs encountered: {}")

    # Deduplicate display rows for alias-merged classes (multiple source names -> one global ID).
    by_global_id: dict[int, TargetClass] = {}
    for target in sorted(mappings, key=lambda item: item.global_id):
        by_global_id[target.global_id] = target

    print("Per-target image counts:")
    for global_id in sorted(by_global_id):
        target = by_global_id[global_id]
        print(f"  {global_id} ({target.folder_name}): {class_image_counts[global_id]}")

    print("Per-target box counts:")
    for global_id in sorted(by_global_id):
        target = by_global_id[global_id]
        print(f"  {global_id} ({target.folder_name}): {class_box_counts[global_id]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Route staged Open Images labels into per-class folders with schema IDs"
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path.home() / "Documents/YOLO_Training_Data/openimages_raw",
        help="Path containing images/, labels/, classes.txt from Open Images export",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.home() / "Documents/YOLO_Training_Data/sorted_images_openimages",
        help="Root where per-class folders will be written",
    )
    parser.add_argument(
        "--exclude-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV of images to exclude by UID. "
            "UID matching is substring-based against filename stems, so "
            "train_<uid>.jpg and train_<uid>.txt are both excluded. "
            "If omitted, the script auto-uses <staging-dir>/exclude.csv when present."
        ),
    )
    parser.add_argument(
        "--source-class",
        action="append",
        default=[],
        help=(
            "Open Images class name. The script resolves global ID from "
            "config/custom_class_map.json and auto-generates folder name '<id>_<canonical_name>'. "
            "Repeat this flag for multiple classes."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write files. Without this flag the script runs a dry run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwrite when outputs already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exclude_csv = args.exclude_csv
    if exclude_csv is None:
        candidate = args.staging_dir / "exclude.csv"
        if candidate.exists():
            exclude_csv = candidate

    excluded_uids = load_excluded_uids(exclude_csv)
    if exclude_csv is not None:
        print(f"Loaded {len(excluded_uids)} excluded UIDs from {exclude_csv}")

    if args.source_class:
        mappings = build_targets_from_source_classes(args.source_class)
    else:
        staging_classes = load_staging_classes(args.staging_dir / "classes.txt")
        mappings, unresolved = build_targets_from_staging_classes(staging_classes)
        print("No --source-class provided; using all staging classes resolvable in canonical map.")
        if unresolved:
            print(f"Unresolved staging classes (skipped): {unresolved}")

        if not mappings:
            raise ValueError(
                "No staging classes could be resolved in custom_class_map.json. "
                "Provide --source-class or update class aliases."
            )

    # Validate duplicate global IDs or source names in provided mapping.
    source_names = [m.source_name for m in mappings]
    if len(source_names) != len(set(source_names)):
        raise ValueError("Duplicate source class names found in --source-class entries")

    route(
        staging_dir=args.staging_dir,
        output_root=args.output_root,
        mappings=mappings,
        apply=args.apply,
        overwrite=args.overwrite,
        excluded_uids=excluded_uids,
    )


if __name__ == "__main__":
    main()
