#!/usr/bin/env python3
"""Compare label filenames between live labels and test_output labels per dataset folder."""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

YOLO_ROOT = "/Users/michaelmandiberg/Documents/YOLO_Training_Data/sorted_images_reprocess/"
WALK_FOLDERS = True
SUBFOLDER_PATH = "test_output/"
SHOW_ID_MATCHES = True
VERBOSE = False
WRITE_NEW_LABELS = True
DRY_RUN = False


def display_path(path: Path) -> str:
    root = Path(YOLO_ROOT)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def is_hidden_name(name: str) -> bool:
    return name.startswith(".")


def discover_dataset_folders(yolo_root: Path) -> list[Path]:
    dataset_folders: list[Path] = []
    for name in sorted(os.listdir(yolo_root)):
        if is_hidden_name(name):
            continue

        dataset_path = yolo_root / name
        if not dataset_path.is_dir():
            continue

        live_labels = dataset_path / "labels"
        live_images = dataset_path / "images"
        new_labels = dataset_path / SUBFOLDER_PATH / "labels" if SUBFOLDER_PATH else dataset_path / "labels"
        if new_labels.is_dir() and (live_labels.is_dir() or live_images.is_dir()):
            dataset_folders.append(dataset_path)
        else:
            print(
                "Skipping "
                f"{display_path(dataset_path)}: expected {display_path(new_labels)} and at least one of "
                f"{display_path(live_labels)} or {display_path(live_images)}"
            )

    return dataset_folders


def collect_label_filenames(labels_dir: Path) -> set[str]:
    return {path.name for path in labels_dir.glob("*.txt")}


def collect_image_filenames(images_dir: Path) -> set[str]:
    return {path.name for path in images_dir.iterdir() if path.is_file()}


def extract_image_id(filename: str) -> str | None:
    stem = Path(filename).stem
    # Prefer realistic image IDs: use the last numeric group with length >= 5.
    # This handles patterns like:
    #   118194051.txt -> 118194051
    #   021a3cd2-0.60_50834295.txt -> 50834295
    #   0.01_118605281_YOLO_debug.txt -> 118605281
    numeric_groups = re.findall(r"\d+", stem)
    if not numeric_groups:
        return None

    long_groups = [group for group in numeric_groups if len(group) >= 5]
    if long_groups:
        return long_groups[-1]

    # Fallback for short numeric IDs.
    return numeric_groups[-1]


def build_image_id_index(filenames: set[str]) -> tuple[dict[str, list[str]], list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    no_id: list[str] = []

    for name in sorted(filenames):
        image_id = extract_image_id(name)
        if image_id is None:
            no_id.append(name)
            continue
        index[image_id].append(name)

    return dict(index), no_id


def compare_folder(dataset_dir: Path) -> dict[str, object]:
    live_labels = dataset_dir / "labels"
    live_images = dataset_dir / "images"
    new_labels = dataset_dir / SUBFOLDER_PATH / "labels" if SUBFOLDER_PATH else dataset_dir / "labels"

    live_names = collect_label_filenames(live_labels) if live_labels.is_dir() else set()
    live_image_names = collect_image_filenames(live_images) if live_images.is_dir() else set()
    new_names = collect_label_filenames(new_labels) if new_labels.is_dir() else set()
    live_image_stems = {Path(name).stem for name in live_image_names}

    same = live_names & new_names
    only_live = live_names - new_names
    only_new = new_names - live_names

    live_index, live_no_id = build_image_id_index(live_names)
    live_image_index, live_image_no_id = build_image_id_index(live_image_names)

    matched_pairs: list[tuple[str, str]] = []
    matched_image_pairs: list[tuple[str, str]] = []
    ambiguous_pairs: list[tuple[str, list[str]]] = []
    ambiguous_image_pairs: list[tuple[str, list[str]]] = []
    not_found: list[str] = []
    no_id_in_new: list[str] = []
    write_operations: list[tuple[Path, Path, str]] = []
    missing_live_image_for_write: list[tuple[str, str]] = []

    for new_name in sorted(new_names):
        image_id = extract_image_id(new_name)
        if image_id is None:
            no_id_in_new.append(new_name)
            continue

        live_matches = live_index.get(image_id, [])
        if not live_matches:
            image_matches = live_image_index.get(image_id, [])
            if not image_matches:
                not_found.append(new_name)
                continue

            if len(image_matches) == 1:
                matched_image_pairs.append((image_matches[0], new_name))
                source_path = new_labels / new_name
                target_filename = f"{Path(image_matches[0]).stem}.txt"
                target_path = live_labels / target_filename
                write_operations.append((source_path, target_path, "matched_from_live_image"))
                continue

            ambiguous_image_pairs.append((new_name, image_matches))
            continue

        if len(live_matches) == 1:
            matched_pairs.append((live_matches[0], new_name))
            source_path = new_labels / new_name
            target_path = live_labels / live_matches[0]
            target_stem = Path(live_matches[0]).stem
            if target_stem in live_image_stems:
                write_operations.append((source_path, target_path, "matched_from_live_label"))
            else:
                missing_live_image_for_write.append((new_name, live_matches[0]))
            continue

        ambiguous_pairs.append((new_name, live_matches))

    live_duplicate_ids = {image_id: names for image_id, names in live_index.items() if len(names) > 1}

    return {
        "same": len(same),
        "different": len(only_live) + len(only_new),
        "only_live": len(only_live),
        "only_new": len(only_new),
        "same_names": sorted(same),
        "only_live_names": sorted(only_live),
        "only_new_names": sorted(only_new),
        "matched_pairs": matched_pairs,
        "matched_image_pairs": matched_image_pairs,
        "ambiguous_pairs": ambiguous_pairs,
        "ambiguous_image_pairs": ambiguous_image_pairs,
        "not_found": not_found,
        "no_id_in_new": no_id_in_new,
        "live_duplicate_ids": live_duplicate_ids,
        "live_no_id": live_no_id,
        "live_image_no_id": live_image_no_id,
        "live_images_total": len(live_image_names),
        "write_operations": write_operations,
        "missing_live_image_for_write": missing_live_image_for_write,
    }


def execute_write_operations(dataset_name: str, stats: dict[str, object]) -> None:
    write_operations = stats["write_operations"]
    missing_live_image_for_write = stats["missing_live_image_for_write"]

    if missing_live_image_for_write:
        print("\033[91mXX XX XX ERROR: Missing live image for matched label target(s) XX XX XX\033[0m")
        print(f"  dataset={dataset_name} count={len(missing_live_image_for_write)}")
        for new_name, live_label_name in missing_live_image_for_write[:10]:
            print(f"    XX no live image for target label {live_label_name} (from new {new_name})")

    if not write_operations:
        print(f"  write plan for {dataset_name}: no eligible writes")
        return

    mode = "DRY RUN" if DRY_RUN else "WRITE"
    print(f"  write plan for {dataset_name} ({mode}): {len(write_operations)} file(s)")

    written = 0
    skipped_missing_source = 0
    for source_path, target_path, reason in write_operations:
        if DRY_RUN:
            print(f"    {display_path(source_path)} -> {display_path(target_path)} [{reason}]")
            continue

        if not source_path.exists():
            skipped_missing_source += 1
            print(f"    missing source, skipping: {display_path(source_path)}")
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(source_path.read_text())
        written += 1

    if not DRY_RUN:
        print(f"    wrote={written} skipped_missing_source={skipped_missing_source}")


def print_samples(dataset_name: str, stats: dict[str, object], limit: int = 10) -> None:
    if VERBOSE: print(f"  samples for {dataset_name}:")
    groups = [
        ("same", stats["same_names"]),
        ("only_live", stats["only_live_names"]),
        ("only_new", stats["only_new_names"]),
    ]

    for label, names in groups:
        sample = names[:limit]
        print(f"    {label} ({len(sample)}/{len(names)}):")
        if not sample:
            print("      -")
            continue
        for name in sample:
            print(f"      {name}")


def print_image_id_match_report(dataset_name: str, stats: dict[str, object], limit: int = 10) -> None:
    matched_pairs = stats["matched_pairs"]
    matched_image_pairs = stats["matched_image_pairs"]
    ambiguous_pairs = stats["ambiguous_pairs"]
    ambiguous_image_pairs = stats["ambiguous_image_pairs"]
    not_found = stats["not_found"]
    no_id_in_new = stats["no_id_in_new"]
    live_duplicate_ids = stats["live_duplicate_ids"]

    print(f"  image_id match summary for {dataset_name}:")
    print(f"    matched_unique_from_live_labels={len(matched_pairs)}")
    print(f"    matched_unique_from_live_images={len(matched_image_pairs)}")
    print(f"    matched_total={len(matched_pairs) + len(matched_image_pairs)}")
    print(f"    not_found={len(not_found)}")
    print(f"    no_image_id_in_new={len(no_id_in_new)}")
    print(f"    live_images_total={stats['live_images_total']}")

    cannot_find = [f"no_live_match: {name}" for name in not_found]
    cannot_find.extend(f"no_image_id: {name}" for name in no_id_in_new)
    print(f"  cannot-find count={len(cannot_find)}")
    if VERBOSE: print(f"  first {limit} cannot-find examples:")
    if not cannot_find:
        print("    -")
    else:
        for entry in cannot_find[:limit]:
            print(f"    {entry}")

    if not VERBOSE:
        return

    print(f"    ambiguous_multiple_live_labels={len(ambiguous_pairs)}")
    print(f"    ambiguous_multiple_live_images={len(ambiguous_image_pairs)}")
    print(f"    live_image_ids_with_duplicates={len(live_duplicate_ids)}")

    print(f"  first {limit} matched live_filename - new_filename:")
    combined_matches = [("label", live_name, new_name) for live_name, new_name in matched_pairs]
    combined_matches.extend(("image", live_name, new_name) for live_name, new_name in matched_image_pairs)
    if not combined_matches:
        print("    -")
    else:
        for source, live_name, new_name in combined_matches[:limit]:
            print(f"    [{source}] {live_name} - {new_name}")

    if ambiguous_pairs:
        print(f"  first {limit} ambiguous label examples (new -> live label candidates):")
        for new_name, live_candidates in ambiguous_pairs[:limit]:
            print(f"    {new_name} -> {', '.join(live_candidates[:3])}")

    if ambiguous_image_pairs:
        print(f"  first {limit} ambiguous image examples (new -> live image candidates):")
        for new_name, live_candidates in ambiguous_image_pairs[:limit]:
            print(f"    {new_name} -> {', '.join(live_candidates[:3])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare label filenames between live and test_output labels")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(YOLO_ROOT),
        help="Root directory containing per-class folders",
    )
    parser.add_argument(
        "--show-samples",
        action="store_true",
        help="Print up to 10 filenames each for same, only_live, and only_new per folder.",
    )
    parser.add_argument(
        "--show-id-matches",
        dest="show_id_matches",
        action="store_true",
        help="Print image_id-based matching report with first 10 matched and first 10 cannot-find examples.",
    )
    parser.add_argument(
        "--no-show-id-matches",
        dest="show_id_matches",
        action="store_false",
        help="Disable image_id-based matching report output.",
    )
    parser.set_defaults(show_id_matches=SHOW_ID_MATCHES)

    args = parser.parse_args()

    if WALK_FOLDERS:
        yolo_root = args.base_dir
        if not yolo_root.is_dir():
            raise FileNotFoundError(f"YOLO_ROOT not found: {yolo_root}")

        dataset_bases = discover_dataset_folders(yolo_root)
        if not dataset_bases:
            print("No valid dataset folders found.")
            return

        for dataset_dir in dataset_bases:
            stats = compare_folder(dataset_dir)
            if VERBOSE: print(
                f"{dataset_dir.name}: same={stats['same']} different={stats['different']} "
                f"(only_live={stats['only_live']} only_new={stats['only_new']})"
            )
            if args.show_samples and VERBOSE:
                print_samples(dataset_dir.name, stats)
            if args.show_id_matches:
                print_image_id_match_report(dataset_dir.name, stats)
            if WRITE_NEW_LABELS:
                execute_write_operations(dataset_dir.name, stats)
    else:
        stats = compare_folder(args.base_dir)
        print(
            f"{args.base_dir.name}: same={stats['same']} different={stats['different']} "
            f"(only_live={stats['only_live']} only_new={stats['only_new']})"
        )
        if args.show_samples and VERBOSE:
            print_samples(args.base_dir.name, stats)
        if args.show_id_matches:
            print_image_id_match_report(args.base_dir.name, stats)
        if WRITE_NEW_LABELS:
            execute_write_operations(args.base_dir.name, stats)


if __name__ == "__main__":
    main()
