import argparse
from pathlib import Path


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def polygon_to_bbox(parts: list[str]) -> str | None:
    """
    Convert a YOLO polygon-style line with 9 elements:
      class x1 y1 x2 y2 x3 y3 x4 y4
    into YOLO bbox line with 5 elements:
      class x_center y_center width height
    """
    if len(parts) != 9:
        return None

    class_id = parts[0]

    try:
        coords = [float(v) for v in parts[1:]]
    except ValueError:
        return None

    xs = coords[0::2]
    ys = coords[1::2]

    x_min = clamp01(min(xs))
    x_max = clamp01(max(xs))
    y_min = clamp01(min(ys))
    y_max = clamp01(max(ys))

    width = clamp01(x_max - x_min)
    height = clamp01(y_max - y_min)
    x_center = clamp01(x_min + (width / 2.0))
    y_center = clamp01(y_min + (height / 2.0))

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def convert_label_file(path: Path, overwrite: bool) -> tuple[int, int, int, bool]:
    """
    Returns: (converted_count, kept_5_count, skipped_count, changed)
    """
    converted_count = 0
    kept_5_count = 0
    skipped_count = 0
    changed = False

    lines = path.read_text(encoding="utf-8").splitlines()
    out_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()

        if len(parts) == 5:
            out_lines.append(stripped)
            kept_5_count += 1
            continue

        converted = polygon_to_bbox(parts)
        if converted is not None:
            out_lines.append(converted)
            converted_count += 1
            changed = True
            continue

        skipped_count += 1

    if changed and overwrite:
        path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

    return converted_count, kept_5_count, skipped_count, changed


def find_label_files(folder: Path) -> list[Path]:
    labels_dir = folder / "labels"
    if labels_dir.is_dir():
        return sorted(labels_dir.glob("*.txt"))

    return sorted(folder.glob("*.txt"))


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO 9-value polygon annotations into 5-value bbox annotations"
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Folder containing labels/ subfolder, or a labels folder itself",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files",
    )
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    label_files = find_label_files(folder)
    if not label_files:
        print(f"No label .txt files found in: {folder}")
        return

    overwrite = not args.dry_run

    total_files_changed = 0
    total_converted = 0
    total_kept_5 = 0
    total_skipped = 0

    for label_file in label_files:
        converted, kept_5, skipped, changed = convert_label_file(label_file, overwrite)
        total_converted += converted
        total_kept_5 += kept_5
        total_skipped += skipped
        if changed:
            total_files_changed += 1

    print("=== Conversion Summary ===")
    print(f"Folder: {folder}")
    print(f"Label files scanned: {len(label_files)}")
    print(f"Files changed: {total_files_changed}")
    print(f"9->5 lines converted: {total_converted}")
    print(f"Already 5-element lines kept: {total_kept_5}")
    print(f"Skipped invalid lines: {total_skipped}")
    print(f"Mode: {'dry-run' if args.dry_run else 'overwrite'}")


if __name__ == "__main__":
    main()
