#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
DEFAULT_SEARCH_ROOT = Path("/Users/michael.mandiberg/Documents/YOLO_Training_Data")
DEFAULT_EXCLUDED_FOLDERS = ("depricated", "reprocess")


def parse_yolo_line(line: str):
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(float(parts[0]))
        x_center, y_center, width, height = map(float, parts[1:])
        area = max(0.0, width) * max(0.0, height)
        return {
            "class_id": class_id,
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
            "area": area,
            "raw": line.rstrip("\n"),
        }
    except ValueError:
        return None


def gather_image_stems(images_detections_dir: Path) -> list[str]:
    stems = {
        path.stem
        for path in images_detections_dir.rglob("*")
        if path.is_file() and path.suffix in IMAGE_EXTS
    }
    return sorted(stems)


def is_excluded_path(path: Path, excluded_folder_names: set[str]) -> bool:
    return any(part in excluded_folder_names for part in path.parts)


def build_label_index(search_root: Path, excluded_folder_names: set[str]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for txt_path in search_root.rglob("*.txt"):
        if is_excluded_path(txt_path, excluded_folder_names):
            continue
        index.setdefault(txt_path.stem, []).append(txt_path)

    for stem in index:
        index[stem].sort()
    return index


def inspect_labels(images_detections_dir: Path, labels_dir: Path, limit: int | None = None) -> dict:
    stems = gather_image_stems(images_detections_dir)
    if limit is not None:
        stems = stems[:limit]

    found_labels = 0
    missing_labels = 0
    modified_labels = 0

    for stem in stems:
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.exists():
            print(f"\n[MISSING LABEL] {label_path}")
            missing_labels += 1
            continue

        lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        parsed = []
        for idx, line in enumerate(lines):
            data = parse_yolo_line(line)
            if data is not None:
                data["idx"] = idx
                parsed.append(data)

        suspect_idx = None
        if parsed:
            smallest = min(parsed, key=lambda row: row["area"])
            suspect_idx = smallest["idx"]

        print(f"\n=== {label_path} ===")
        if not lines:
            print("(empty label file)")
        for idx, line in enumerate(lines):
            marker = " <--" if suspect_idx is not None and idx == suspect_idx else ""
            print(f"{line}{marker}")

        if len(parsed) >= 2:
            parsed_sorted = sorted(parsed, key=lambda row: row["area"])
            smallest_area = parsed_sorted[0]["area"]
            next_area = parsed_sorted[1]["area"]
            ratio = smallest_area / next_area if next_area > 0 else 0.0
            print(f"[stats] smallest_area={smallest_area:.6f} next_area={next_area:.6f} ratio={ratio:.6f}")

        found_labels += 1

    return {
        "stems_scanned": len(stems),
        "labels_found": found_labels,
        "labels_missing": missing_labels,
        "labels_modified": modified_labels,
    }


def delete_suspect_lines(
    images_detections_dir: Path,
    labels_dir: Path,
    search_root: Path | None = None,
    exclude_folders: tuple[str, ...] = DEFAULT_EXCLUDED_FOLDERS,
    limit: int | None = None,
) -> dict:
    stems = gather_image_stems(images_detections_dir)
    if limit is not None:
        stems = stems[:limit]

    excluded_folder_names = set(exclude_folders)
    label_index: dict[str, list[Path]] = {}
    if search_root is not None:
        label_index = build_label_index(search_root=search_root, excluded_folder_names=excluded_folder_names)

    found_labels = 0
    missing_labels = 0
    modified_labels = 0
    skipped_single_or_empty = 0
    ambiguous_labels = 0
    missing_label_paths: list[str] = []
    ambiguous_label_groups: dict[str, list[str]] = {}

    for stem in stems:
        label_path: Path | None = None
        if search_root is not None:
            candidates = label_index.get(stem, [])
            if not candidates:
                expected_path = f"{search_root}/**/{stem}.txt"
                print(f"\n[MISSING LABEL] {expected_path} (excluding folders: {sorted(excluded_folder_names)})")
                missing_label_paths.append(expected_path)
                missing_labels += 1
                continue
            if len(candidates) > 1:
                print(f"\n[AMBIGUOUS LABEL] stem '{stem}' has multiple matches:")
                for path in candidates:
                    print(f"- {path}")
                ambiguous_label_groups[stem] = [str(path) for path in candidates]
                ambiguous_labels += 1
                continue
            label_path = candidates[0]
        else:
            fallback_path = labels_dir / f"{stem}.txt"
            if not fallback_path.exists():
                print(f"\n[MISSING LABEL] {fallback_path}")
                missing_label_paths.append(str(fallback_path))
                missing_labels += 1
                continue
            label_path = fallback_path

        lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        parsed = []
        for idx, line in enumerate(lines):
            data = parse_yolo_line(line)
            if data is not None:
                data["idx"] = idx
                parsed.append(data)

        if len(parsed) < 2:
            print(f"\n[SKIP] {label_path} has <2 valid annotations; no deletion")
            skipped_single_or_empty += 1
            found_labels += 1
            continue

        suspect = min(parsed, key=lambda row: row["area"])
        suspect_idx = suspect["idx"]

        print(f"\n=== {label_path} ===")
        print(f"Deleting line: {lines[suspect_idx]}")

        new_lines = [line for idx, line in enumerate(lines) if idx != suspect_idx]
        label_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
        modified_labels += 1
        found_labels += 1

    return {
        "stems_scanned": len(stems),
        "labels_found": found_labels,
        "labels_missing": missing_labels,
        "labels_modified": modified_labels,
        "labels_skipped_lt2": skipped_single_or_empty,
        "labels_ambiguous": ambiguous_labels,
        "missing_label_paths": missing_label_paths,
        "ambiguous_label_groups": ambiguous_label_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect or delete suspect lines for labels corresponding to kept incorrect detection images."
    )
    parser.add_argument(
        "--images-detections-dir",
        type=Path,
        default=Path("/Users/michael.mandiberg/Documents/YOLO_Training_Data/reprocess/suspect_images/images_detections"),
        help="Folder containing detection-drawn images that were kept as incorrect.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("/Users/michael.mandiberg/Documents/YOLO_Training_Data/reprocess/suspect_images/labels"),
        help="Folder containing matching label .txt files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of stems to inspect (for testing).",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=DEFAULT_SEARCH_ROOT,
        help="Root folder to recursively find original label files (<stem>.txt).",
    )
    parser.add_argument(
        "--exclude-folders",
        nargs="*",
        default=list(DEFAULT_EXCLUDED_FOLDERS),
        help="Folder names to exclude while searching original labels.",
    )
    parser.add_argument(
        "--apply-delete",
        action="store_true",
        help="If set, delete the flagged suspect line (smallest bbox area) from each matching label file.",
    )

    args = parser.parse_args()

    if not args.images_detections_dir.exists():
        raise FileNotFoundError(f"images_detections folder not found: {args.images_detections_dir}")
    if not args.labels_dir.exists():
        raise FileNotFoundError(f"labels folder not found: {args.labels_dir}")
    if args.apply_delete and not args.search_root.exists():
        raise FileNotFoundError(f"search root not found: {args.search_root}")

    if args.apply_delete:
        summary = delete_suspect_lines(
            images_detections_dir=args.images_detections_dir,
            labels_dir=args.labels_dir,
            search_root=args.search_root,
            exclude_folders=tuple(args.exclude_folders),
            limit=args.limit,
        )
    else:
        summary = inspect_labels(
            images_detections_dir=args.images_detections_dir,
            labels_dir=args.labels_dir,
            limit=args.limit,
        )

    print("\n=== Summary ===")
    print(summary)
    missing_paths = summary.get("missing_label_paths", [])
    if missing_paths:
        print("\n=== Missing Label Paths ===")
        for path in missing_paths:
            print(path)
    if args.apply_delete:
        print("Suspect lines were deleted from modified files.")
    else:
        print("No files were modified or deleted.")


if __name__ == "__main__":
    main()
