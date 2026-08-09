#!/usr/bin/env python3
"""
Look in each subfolder, and check the labels and images folders for unpaired files.

Usage examples:
# Dry run (default) - just list unpaired files
python remove_unpaired_files.py /path/to/folder

# Delete unpaired files, keeping the latest modified file by mtime (prompt for confirmation)
python remove_unpaired_files.py /path/to/folder --delete --keep latest

# Delete unpaired files without prompting (USE CAREFULLY)
python remove_unpaired_files.py /path/to/folder --delete --force --keep largest
"""
import argparse
import re
from pathlib import Path
from collections import defaultdict
import sys
from typing import Optional

# Built-in constants you can edit
CONST_EXT_GROUPS = {
    "IMAGES": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"],
    "JPEGS": [".jpg", ".jpeg"],
}
# # Mapping of user-friendly constant names to keep policies
# CONST_KEEP_POLICIES = {
#     "FIRST": "first",
#     "LATEST": "latest",
#     "OLDEST": "oldest",
#     "LARGEST": "largest",
# }
# # Defaults (use constant names so you can change them here)
# DEFAULT_KEEP = "FIRST"      # one of keys in CONST_KEEP_POLICIES
# DEFAULT_EXTS = "IMAGES"     # one of keys in CONST_EXT_GROUPS
DEFAULT_FOLDER = Path.cwd()

# # Backward-compatible set of common extensions
# VALID_EXTS = set(CONST_EXT_GROUPS["IMAGES"])

def extract_image_id(filename: str) -> Optional[str]:
    # prefer digits between underscores: e.g. 0.61_118309736_YOLO_debug.jpg
    m = re.search(r'_(\d+)_', filename)
    if m:
        return m.group(1)
    # filename that starts with digits: 1080515528.jpg
    m = re.match(r'^(\d+)(?:\.|$)', Path(filename).name)
    if m:
        return m.group(1)
    # fallback: pick longest digit sequence in the filename
    seqs = re.findall(r'(\d+)', filename)
    if seqs:
        return max(seqs, key=len)
    return None

def remove_unpaired_files(images_folder: Path, labels_folder: Path, dry_run: bool = True):
    """
    Remove unpaired files between images and labels folders.

    Args:
        images_folder (Path): Path to the images folder.
        labels_folder (Path): Path to the labels folder.
        dry_run (bool): If True, only list unpaired files without deleting.
        delete (bool): If True, delete unpaired files.
        force (bool): If True, skip confirmation prompt when deleting.
        keep_policy (str): Policy for which file to keep in each unpaired group.
        report_path (Optional[Path]): Optional path to write a summary report.
    """
    # load all files in images and labels folders with the specified extensions
    images = {f for f in images_folder.iterdir() if f.is_file() and f.suffix.lower() == ".jpg" or f.suffix.lower() == ".jpeg" or f.suffix.lower() == ".png"}
    labels = {f for f in labels_folder.iterdir() if f.is_file() and f.suffix.lower() == ".txt"}

    # find unpaired images and labels
    unpaired_images = {f for f in images if not (labels_folder / f.with_suffix('.txt').name).exists()}
    unpaired_labels = {f for f in labels if not (images_folder / f.with_suffix('.jpg').name).exists() and not (images_folder / f.with_suffix('.jpeg').name).exists() and not (images_folder / f.with_suffix('.png').name).exists()} 

    # report the findings
    print(f"Found {len(unpaired_images)} unpaired images and {len(unpaired_labels)} unpaired labels in {images_folder.parent}") 

    if not dry_run:
        # delete unpaired files
        for f in unpaired_images:
            print(f"Deleting unpaired image: {f}")
            f.unlink()
        for f in unpaired_labels:
            print(f"Deleting unpaired label: {f}")
            f.unlink()

    
def main():
    p = argparse.ArgumentParser(description="Remove unpaired files between images and labels folders.")
    p.add_argument("folder", nargs='?', type=Path, default=DEFAULT_FOLDER,
                   help=f"Folder containing images and labels (default: {DEFAULT_FOLDER})")
    p.add_argument("--exts", nargs="+", help="File extensions to consider (default: IMAGES group). You can pass a constant name (e.g., IMAGES) or a list of extensions (e.g., .jpg .png).", default=None)
    p.add_argument("--dry-run", action="store_true", help="Only list unpaired files (default is dry-run).")
    p.add_argument("--delete", action="store_true", help="Delete unpaired files. Requires confirmation unless --force.")
    p.add_argument("--force", action="store_true", help="When combined with --delete, don't prompt; proceed directly.")
    # --keep accepts either a policy name (first/latest/oldest/largest) or a constant key from CONST_KEEP_POLICIES (e.g., LATEST)
    # p.add_argument("--keep", choices=list(CONST_KEEP_POLICIES.values()) + list(CONST_KEEP_POLICIES.keys()), default=DEFAULT_KEEP,
    #                help=f"Which file to keep in each unpaired  group. You can pass a policy name (first/latest/oldest/largest) or a constant like LATEST. (Default: {DEFAULT_KEEP})")
    p.add_argument("--report", type=Path, help="Optional CSV path to write a summary report.")
    args = p.parse_args()

    folder: Path = args.folder
    if not folder.exists() or not folder.is_dir():
        print("Error: folder does not exist or is not a directory.", file=sys.stderr)
        sys.exit(2)

    # # normalize extensions to start with '.' and be lowercase
    # exts = {e if e.startswith(".") else f".{e}" for e in exts}
    # exts = {e.lower() for e in exts}


    # load all folders in folder, and check for 'images' and 'labels' subfolders
    subfolders = [f for f in folder.iterdir() if f.is_dir()]
    if not subfolders:
        print(f"No subfolders found in {folder}. Exiting.", file=sys.stderr)
        sys.exit(3)
    for subfolder in subfolders:
        images_folder = subfolder / "images"
        labels_folder = subfolder / "labels"
        if not images_folder.exists() or not labels_folder.exists():
            print(f"Skipping {subfolder}: missing 'images' or 'labels' folder.", file=sys.stderr)
            continue
        print(f"Processing subfolder: {subfolder}")
        # call the function to remove unpaired files
        remove_unpaired_files(images_folder, labels_folder, args.dry_run)

if __name__ == "__main__":
    main()