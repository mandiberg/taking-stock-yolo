import os
import shutil
from pathlib import Path

ROOT = "/Volumes/LaCie/segment_images_84_valentine/"
FOLDERS = ["test_output_noblems/84", "test_output_blems/84"]


def strip_conf_score(filename):
    """
    Remove the confidence score prefix from filename.
    E.g., "0.40_10194797_YOLO_debug.jpg" -> "10194797_YOLO_debug.jpg"
    """
    # Remove extension
    name, ext = os.path.splitext(filename)
    
    # Split by underscore and check if first part is a conf score (D.DD format)
    parts = name.split('_', 1)
    if len(parts) >= 2:
        first_part = parts[0]
        try:
            # Try to parse as float; if successful, it's the conf score
            float(first_part)
            # Reconstruct without the score
            return parts[1] + ext
        except ValueError:
            # Not a float, return original
            return filename
    return filename


def get_jpg_basenames(folder_path):
    """Get set of JPG basenames (without extension) in a folder, with conf score stripped"""
    jpg_files = []
    for f in os.listdir(folder_path):
        if f.lower().endswith(('.jpg', '.jpeg')):
            jpg_files.append(f)
    return jpg_files


def move_unique_files(folder_path, basenames, subfolder="unique"):
    """Move files with given basenames to a subfolder"""
    unique_dir = os.path.join(folder_path, subfolder)
    os.makedirs(unique_dir, exist_ok=True)
    
    moved_count = 0
    for basename in basenames:
        src_path = os.path.join(folder_path, basename)
        if os.path.isfile(src_path):
            dst_path = os.path.join(unique_dir, basename)
            try:
                shutil.move(src_path, dst_path)
                moved_count += 1
            except Exception as e:
                print(f"[WARN] Could not move {basename}: {e}")
    
    return moved_count


def main():
    # Validate folders exist
    folder_paths = {}
    for folder_name in FOLDERS:
        full_path = os.path.join(ROOT, folder_name)
        if not os.path.isdir(full_path):
            print(f"Folder not found: {full_path}")
            return
        folder_paths[folder_name] = full_path
    
    # Collect JPG basenames from each folder
    print(f"Scanning folders for JPG files...\n")
    
    file_sets = {}
    file_mapping = {}  # Map stripped name -> original full filename per folder
    
    for folder_name, folder_path in folder_paths.items():
        jpg_files = get_jpg_basenames(folder_path)
        stripped_files = {}
        
        for original_filename in jpg_files:
            stripped_name = strip_conf_score(original_filename)
            stripped_files[stripped_name] = original_filename
        
        file_sets[folder_name] = set(stripped_files.keys())
        file_mapping[folder_name] = stripped_files
        print(f"{folder_name}: {len(jpg_files)} JPG files ({len(stripped_files)} unique after stripping conf score)")
    
    # Compare sets (using stripped names)
    print(f"\nComparing sets (ignoring confidence scores)...")
    
    set_blems = file_sets[FOLDERS[1]]  # test_output_blems
    set_noblems = file_sets[FOLDERS[0]]  # test_output_noblems
    
    # Set operations
    unique_to_blems = set_blems - set_noblems
    unique_to_noblems = set_noblems - set_blems
    common = set_blems & set_noblems
    
    print(f"  Common to both: {len(common)}")
    print(f"  Unique to {FOLDERS[1]}: {len(unique_to_blems)}")
    print(f"  Unique to {FOLDERS[0]}: {len(unique_to_noblems)}")
    
    # Move unique files (using original filenames)
    print(f"\nMoving unique files...")
    
    if len(unique_to_noblems) > 0:
        noblems_path = folder_paths[FOLDERS[0]]
        noblems_mapping = file_mapping[FOLDERS[0]]
        original_filenames = {noblems_mapping[stripped] for stripped in unique_to_noblems}
        moved = move_unique_files(noblems_path, original_filenames)
        print(f"  Moved {moved} files from {FOLDERS[0]} to {FOLDERS[0]}/unique")
    
    if len(unique_to_blems) > 0:
        blems_path = folder_paths[FOLDERS[1]]
        blems_mapping = file_mapping[FOLDERS[1]]
        original_filenames = {blems_mapping[stripped] for stripped in unique_to_blems}
        moved = move_unique_files(blems_path, original_filenames)
        print(f"  Moved {moved} files from {FOLDERS[1]} to {FOLDERS[1]}/unique")
    
    print(f"\nDone.")


if __name__ == "__main__":
    main()
