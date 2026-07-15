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


FOLDER = "/Users/michaelmandiberg/Documents/YOLO_Training_Data/sorted_images_orig/misc_balls1"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def extract_uid(filename: str) -> str | None:
    # print(f"extract_uid: filename='{filename}'")
    basename = filename.split(".")[0]  # Remove extension
    parts = basename.split("_", 1)
    # print(f"extract_uid: parts={parts}")
    if len(parts) < 2:
        if "-" in parts[0]:
            sub_parts = parts[0].split("-", 1)
            for part in sub_parts:
                if part and part.isdigit():
                    # print(f"extract_uid: filename='{filename}' - found UID '{part}'")
                    return part
            for part in sub_parts:
                if part and any(c.isalnum() for c in part):
                    # print(f"extract_uid: filename='{filename}' - found UID '{part}'")
                    return part
        elif len(parts) == 1 and parts[0].isdigit():
            # print(f"extract_uid: filename='{filename}' - found UID '{parts[0]}'")
            return parts[0]
        else:
            # print(f"extract_uid: filename='{filename}' - not enough parts")
            return None
    candidate = parts[0]
    if "-" in candidate:
        print(f"extract_uid: filename='{filename}' - returning entire filename '{candidate}'")
        return candidate
    # print(f"extract_uid: filename='{filename}' - no UID found")
    return None


def strip_uid_prefix(filename: str) -> str:
    uid = extract_uid(filename)
    if uid is None:
        return filename
    print(f"strip_uid_prefix: filename='{filename}' - UID found '{uid}'")
    if "-" in filename:
        return filename.split("-", 1)[0]
    else:
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
      0416be30-171905043-minimal-portrait-of-mixed-race-teenage-boy-holding-basketbal_TSJqsYj.txt -> 171905043-minimal-portrait-of-mixed-race-teenage-boy-holding-basketbal_TSJqsYj
    """
    name_no_ext = Path(filename).stem
    base = strip_uid_prefix(name_no_ext)
    print(f"normalize_match_key: filename='{filename}' - base='{base}'")
    parts = base.split("_", 1)
    if len(parts) == 2:
        print(f"normalize_match_key: base='{base}' from filename='{filename}'")
        first, rest = parts
        if len(first) > len(rest):
            print(f"Warning: first part '{first}' is longer than rest '{rest}' in filename '{filename}'")
            # this is the case where filename was trimmed and a replace with _TSJqsYj etc
            # in this case, we want to to return the whole thing I think? or maybe just part 0
            return base
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
    print(f"extract_image_id: base='{base}' from filename='{filename}'")
    if "_" in base:
        parts = base.split("_", 1)
    elif "-" in base:
        parts = base.split("-", 1)
    else:
        parts = [base]
    for part in parts:
        # first only digits
        if part.isdigit():
            return part
    for part in parts:
        # if that doesn't work, try to find a part that contains a digit and return that
        for char in part:
            if char.isdigit():
                return part
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