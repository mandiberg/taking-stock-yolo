#!/usr/bin/env python3
"""
Merge multiple flower dataset folders into a single unified dataset.

Input structure:
  /YOLO_Training_Data/flowers_individual/
    97_rose/
      images/
      labels/
    98_tulip/
      images/
      labels/
    ...

Output structure:
  /YOLO_Training_Data/flowers_individual/
    101_flowers/
      images/
      labels/

Class ID remapping:
  - Class IDs 97-103 (all flower types) -> remapped to 101
  - Other class IDs -> preserved as-is
"""

import os
import shutil
from pathlib import Path


def process_flower_datasets(source_root, output_folder_name="101_flowers"):
    """
    Merge all flower datasets from subfolders into a single folder.
    
    Args:
        source_root: Path to YOLO_Training_Data/flowers_individual
        output_folder_name: Name of output folder (default: 101_flowers)
    """
    
    source_root = Path(source_root)
    output_root = source_root / output_folder_name
    output_images = output_root / "images"
    output_labels = output_root / "labels"
    
    # Create output directories
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_root}")
    print(f"Images will be saved to: {output_images}")
    print(f"Labels will be saved to: {output_labels}\n")
    
    # Find all flower class folders (97_*, 98_*, etc.)
    flower_folders = []
    for item in source_root.iterdir():
        if item.is_dir() and not item.name.startswith('101'):
            # Check if it has images and labels subdirs
            if (item / "images").exists() and (item / "labels").exists():
                flower_folders.append(item)
    
    flower_folders.sort()
    
    if not flower_folders:
        print("No flower class folders found!")
        return
    
    print(f"Found {len(flower_folders)} flower class folders:")
    for folder in flower_folders:
        print(f"  - {folder.name}")
    print()
    
    # Statistics
    total_images = 0
    total_labels = 0
    remapped_lines = 0
    preserved_lines = 0
    skipped_images = 0
    
    # Process each flower class folder
    for class_folder in flower_folders:
        class_name = class_folder.name
        images_dir = class_folder / "images"
        labels_dir = class_folder / "labels"
        
        print(f"Processing {class_name}...")
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(images_dir.glob(ext))
        
        print(f"  Found {len(image_files)} images")
        
        for img_path in image_files:
            img_name = img_path.name
            img_stem = img_path.stem
            
            # Copy image to output folder
            output_img_path = output_images / img_name
            try:
                shutil.copy2(img_path, output_img_path)
                total_images += 1
            except Exception as e:
                print(f"  Warning: Could not copy image {img_name}: {e}")
                skipped_images += 1
                continue
            
            # Find and process corresponding label file
            label_file = labels_dir / f"{img_stem}.txt"
            
            if not label_file.exists():
                # No label for this image (background image)
                print(f"  No label for {img_name} (background)")
                continue
            
            # Read label file and remap class IDs
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"  Warning: Could not read label {label_file.name}: {e}")
                continue
            
            # Process each line and remap class IDs
            remapped_lines_count = 0
            preserved_lines_count = 0
            converted_lines = []
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    # Invalid line, skip
                    continue
                
                try:
                    class_id = int(parts[0])
                except ValueError:
                    # Invalid class ID, skip line
                    continue
                
                # Check if class ID is in flower range (0-103)
                if 0 <= class_id <= 103:
                    # Remap to 126
                    parts[0] = '126'
                    remapped_lines_count += 1
                    remapped_lines += 1
                else:
                    # Preserve other class IDs
                    preserved_lines_count += 1
                    preserved_lines += 1
                
                # Reconstruct line
                converted_lines.append(' '.join(parts) + '\n')
            
            # Write converted label file
            output_label_path = output_labels / f"{img_stem}.txt"
            try:
                with open(output_label_path, 'w') as f:
                    f.writelines(converted_lines)
                total_labels += 1
                
                if remapped_lines_count > 0 or preserved_lines_count > 0:
                    print(f"  {img_name}: {remapped_lines_count} remapped, {preserved_lines_count} preserved")
            except Exception as e:
                print(f"  Warning: Could not write label {output_label_path.name}: {e}")
        
        print(f"  ✓ {class_name} complete\n")
    
    # Print summary
    print("=" * 70)
    print("MERGE COMPLETE")
    print("=" * 70)
    print(f"Total images copied: {total_images}")
    print(f"Total labels processed: {total_labels}")
    print(f"  - Lines remapped (97-103 → 101): {remapped_lines}")
    print(f"  - Lines preserved (other class IDs): {preserved_lines}")
    print(f"  - Images skipped: {skipped_images}")
    print(f"\nOutput folder: {output_root}")
    print(f"  Images: {output_images}")
    print(f"  Labels: {output_labels}")


if __name__ == '__main__':
    import sys
    
    # Default path
    default_path = "/Users/michaelmandiberg/Documents/YOLO_Training_Data/unify_these"
    
    # Allow custom path as argument
    source_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    if not Path(source_path).exists():
        print(f"Error: Source path does not exist: {source_path}")
        sys.exit(1)
    
    process_flower_datasets(source_path)
