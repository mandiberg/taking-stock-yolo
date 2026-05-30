'''
Compares labels/ and images/ folders and copies UID prefixes from one side to the
other when a matching counterpart is missing the UID.

Example:
labels/0f0febd2-0.79_43660298_YOLO_debug.txt
images/0.79_43660298_YOLO_debug.jpg

Will rename image to:
images/0f0febd2-0.79_43660298_YOLO_debug.jpg
'''

from pathlib import Path


FOLDER = "/Users/michaelmandiberg/Documents/yolo/ClipBoards_Completed"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def extract_uid(filename: str) -> str | None:
    parts = filename.split("_", 1)
    if len(parts) < 2:
        return None
    candidate = parts[0]
    if "-" in candidate:
        return candidate
    return None


def strip_uid_prefix(filename: str) -> str:
    uid = extract_uid(filename)
    if uid is None:
        return filename
    return filename.split("_", 1)[-1]


def normalize_match_key(filename: str) -> str:
    """
    Build a comparable key across label/image names by removing:
    - extension
    - optional UID prefix
    - optional leading confidence prefix like '0.69_'

    Examples:
      0f0febd2-0.79_43660298_YOLO_debug.txt -> 43660298_YOLO_debug
      0.69_43660298_YOLO_debug.jpg         -> 43660298_YOLO_debug
      43660298_YOLO_debug.txt              -> 43660298_YOLO_debug
    """
    name_no_ext = Path(filename).stem
    base = strip_uid_prefix(name_no_ext)

    parts = base.split("_", 1)
    if len(parts) == 2:
        first, rest = parts
        # Only strip confidence-like prefixes such as 0.69_
        if "." in first:
            try:
                conf = float(first)
                if 0.0 <= conf <= 1.0:
                    return rest
            except ValueError:
                pass

    return base


def strip_leading_confidence_token(filename: str) -> str:
    """
    Removes a leading confidence token like '0.70_' from a filename while
    preserving extension.
    """
    p = Path(filename)
    stem = p.stem
    suffix = p.suffix

    parts = stem.split("_", 1)
    if len(parts) == 2:
        first, rest = parts
        if "." in first:
            try:
                conf = float(first)
                if 0.0 <= conf <= 1.0:
                    return f"{rest}{suffix}"
            except ValueError:
                pass

    return filename


def extract_image_id(filename: str) -> str | None:
    base = normalize_match_key(filename)
    image_id = base.split("_", 1)[0]
    if image_id.isdigit():
        return image_id
    return None


def index_by_base(files: list[Path]) -> dict[str, Path]:
    indexed = {}
    for p in files:
        image_id = extract_image_id(p.name)
        if image_id is None:
            print(f"Warning: could not extract image_id from '{p.name}'")
            continue
        if image_id in indexed:
            print(f"Warning: duplicate image_id '{image_id}' in {p.parent}")
            continue
        indexed[image_id] = p
    return indexed


def maybe_rename_with_uid(target_path: Path, uid: str) -> bool:
    if extract_uid(target_path.name) is not None:
        return False

    cleaned_name = strip_leading_confidence_token(target_path.name)
    new_name = f"{uid}_{cleaned_name}"
    new_path = target_path.with_name(new_name)

    if new_path.exists():
        print(f"Cannot rename {target_path.name} -> {new_name} (target exists)")
        return False

    print(f"Renaming {target_path.name} -> {new_name}")
    target_path.rename(new_path)
    return True


def main():
    root = Path(FOLDER)
    labels_dir = root / "labels"
    images_dir = root / "images"

    if not labels_dir.is_dir() or not images_dir.is_dir():
        raise FileNotFoundError(
            f"Expected both labels/ and images/ under {root}"
        )

    label_files = sorted(labels_dir.glob("*.txt"))
    image_files = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    labels_by_base = index_by_base(label_files)
    images_by_base = index_by_base(image_files)
    print(f"Found {len(label_files)} label files and {len(image_files)} image files")
    print(f"Unique base names in labels: {len(labels_by_base)}, images: {len(images_by_base)}")
    print(f"Sample label base names: {list(labels_by_base.keys())[:5]}")
    print(f"Sample image base names: {list(images_by_base.keys())[:5]}")
    shared_bases = sorted(set(labels_by_base.keys()) & set(images_by_base.keys()))

    renamed_count = 0
    for base in shared_bases:
        label_path = labels_by_base[base]
        image_path = images_by_base[base]

        label_uid = extract_uid(label_path.name)
        image_uid = extract_uid(image_path.name)

        if label_uid and not image_uid:
            if maybe_rename_with_uid(image_path, label_uid):
                renamed_count += 1
        elif image_uid and not label_uid:
            if maybe_rename_with_uid(label_path, image_uid):
                renamed_count += 1
        elif label_uid and image_uid and label_uid != image_uid:
            print(
                "UID mismatch for base "
                f"'{base}': label={label_uid}, image={image_uid}"
            )

    print("=== UID Repair Summary ===")
    print(f"labels files: {len(label_files)}")
    print(f"image files: {len(image_files)}")
    print(f"paired base names: {len(shared_bases)}")
    print(f"files renamed: {renamed_count}")


if __name__ == "__main__":
    main()