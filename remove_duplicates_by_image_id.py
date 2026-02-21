#!/usr/bin/env python3
"""
Remove duplicate image files by image_id extracted from filenames.

Usage examples:
# Dry run (default) - just list duplicates
python remove_duplicates_by_image_id.py /path/to/folder

# Delete duplicates, keeping the latest modified file by mtime (prompt for confirmation)
python remove_duplicates_by_image_id.py /path/to/folder --delete --keep latest

# Delete duplicates without prompting (USE CAREFULLY)
python remove_duplicates_by_image_id.py /path/to/folder --delete --force --keep largest
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
# Mapping of user-friendly constant names to keep policies
CONST_KEEP_POLICIES = {
    "FIRST": "first",
    "LATEST": "latest",
    "OLDEST": "oldest",
    "LARGEST": "largest",
}
# Defaults (use constant names so you can change them here)
DEFAULT_KEEP = "FIRST"      # one of keys in CONST_KEEP_POLICIES
DEFAULT_EXTS = "IMAGES"     # one of keys in CONST_EXT_GROUPS
DEFAULT_FOLDER = Path.cwd()

# Backward-compatible set of common extensions
VALID_EXTS = set(CONST_EXT_GROUPS["IMAGES"])

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

def choose_keep_file(files, policy: str):
    if policy == "first":
        return sorted(files, key=lambda p: p.name)[0]
    if policy == "latest":
        return max(files, key=lambda p: p.stat().st_mtime)
    if policy == "oldest":
        return min(files, key=lambda p: p.stat().st_mtime)
    if policy == "largest":
        return max(files, key=lambda p: p.stat().st_size)
    # default fallback
    return sorted(files, key=lambda p: p.name)[0]

def main():
    p = argparse.ArgumentParser(description="Remove duplicate images by image_id parsed from filenames.")
    p.add_argument("folder", nargs='?', type=Path, default=DEFAULT_FOLDER,
                   help=f"Folder containing images (default: {DEFAULT_FOLDER})")
    p.add_argument("--exts", nargs="+", help="File extensions to consider (default: IMAGES group). You can pass a constant name (e.g., IMAGES) or a list of extensions (e.g., .jpg .png).", default=None)
    p.add_argument("--dry-run", action="store_true", help="Only list duplicates (default is dry-run).")
    p.add_argument("--delete", action="store_true", help="Delete duplicates. Requires confirmation unless --force.")
    p.add_argument("--force", action="store_true", help="When combined with --delete, don't prompt; proceed directly.")
    # --keep accepts either a policy name (first/latest/oldest/largest) or a constant key from CONST_KEEP_POLICIES (e.g., LATEST)
    p.add_argument("--keep", choices=list(CONST_KEEP_POLICIES.values()) + list(CONST_KEEP_POLICIES.keys()), default=DEFAULT_KEEP,
                   help=f"Which file to keep in each duplicate group. You can pass a policy name (first/latest/oldest/largest) or a constant like LATEST. (Default: {DEFAULT_KEEP})")
    p.add_argument("--report", type=Path, help="Optional CSV path to write a summary report.")
    args = p.parse_args()

    folder: Path = args.folder
    if not folder.exists() or not folder.is_dir():
        print("Error: folder does not exist or is not a directory.", file=sys.stderr)
        sys.exit(2)

    # Resolve --keep allowing constant names
    raw_keep = args.keep if args.keep is not None else DEFAULT_KEEP
    if isinstance(raw_keep, str) and raw_keep.upper() in CONST_KEEP_POLICIES:
        keep_policy = CONST_KEEP_POLICIES[raw_keep.upper()]
    else:
        keep_policy = str(raw_keep).lower()
    if keep_policy not in set(CONST_KEEP_POLICIES.values()):
        print(f"Error: unknown keep policy '{args.keep}'", file=sys.stderr)
        sys.exit(2)

    # Resolve --exts: allow constant group name (e.g., IMAGES) or explicit list of extensions
    if args.exts:
        if len(args.exts) == 1 and args.exts[0].upper() in CONST_EXT_GROUPS:
            exts = set(CONST_EXT_GROUPS[args.exts[0].upper()])
        else:
            exts = set(args.exts)
    else:
        exts = set(CONST_EXT_GROUPS[DEFAULT_EXTS.upper()])

    # normalize extensions to start with '.' and be lowercase
    exts = {e if e.startswith(".") else f".{e}" for e in exts}
    exts = {e.lower() for e in exts}

    groups = defaultdict(list)
    for f in folder.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() not in exts:
            continue
        img_id = extract_image_id(f.name)
        if img_id:
            groups[img_id].append(f)

    duplicates = {img_id: files for img_id, files in groups.items() if len(files) > 1}
    if not duplicates:
        print("No duplicates found.")
        return

    total_groups = len(duplicates)
    total_files = sum(len(v) for v in duplicates.values())
    print(f"Found {total_groups} image_id groups with duplicates (total {total_files} files).")

    actions = []
    for img_id, files in sorted(duplicates.items()):
        keep = choose_keep_file(files, keep_policy)
        remove = [f for f in files if f != keep]
        print(f"\nimage_id={img_id}: keep -> {keep.name}")
        for r in remove:
            print(f"  duplicate -> {r.name}")
            actions.append((r, img_id, keep))

    if args.report:
        import csv
        with args.report.open("w", newline='', encoding="utf-8") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["image_id", "kept", "removed"])
            by_img = {}
            for r, img_id, keep in actions:
                by_img.setdefault(img_id, {"keep": keep.name, "removed": []})["removed"].append(r.name)
            for img_id, data in by_img.items():
                writer.writerow([img_id, data["keep"], ";".join(data["removed"])])
        print(f"\nReport written to {args.report}")

    if args.delete:
        if not args.force:
            ans = input("\nProceed to delete the listed duplicates? Type YES to confirm: ")
            if ans.strip() != "YES":
                print("Aborted by user.")
                return
        removed_count = 0
        for r, img_id, keep in actions:
            try:
                r.unlink()
                removed_count += 1
            except Exception as e:
                print(f"Failed to remove {r}: {e}", file=sys.stderr)
        print(f"\nDeleted {removed_count} files.")
    else:
        print("\nDry run: no files deleted. Re-run with --delete and optionally --force to remove duplicates.")

if __name__ == "__main__":
    main()