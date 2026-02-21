#!/usr/bin/env python3
"""
Convert Label Studio JSON export to YOLOv8 format.

Label Studio format (normalized percentages):
- x, y: top-left corner (percentage of image width/height)
- width, height: box dimensions (percentage of image width/height)

YOLO format (normalized 0-1):
- class_id x_center y_center width height
- All values normalized to 0-1 range
"""

import json
import os
from pathlib import Path


def convert_labelstudio_to_yolo(json_file, output_dir, class_mapping=None):
    """
    Convert Label Studio JSON export to YOLO format txt files.
    
    Args:
        json_file: Path to Label Studio JSON export
        output_dir: Directory to save YOLO .txt files
        class_mapping: Dict mapping label names to class IDs (e.g., {'Lily': 0, 'Daisy': 1})
                      If None, will auto-generate based on order of appearance
    """
    
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Auto-generate class mapping if not provided
    if class_mapping is None:
        unique_labels = set()
        for task in data:
            if 'annotations' in task:
                for annotation in task['annotations']:
                    if 'result' in annotation:
                        for result in annotation['result']:
                            if 'value' in result and 'rectanglelabels' in result['value']:
                                unique_labels.update(result['value']['rectanglelabels'])
        
        class_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        print(f"Auto-generated class mapping: {class_mapping}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each task
    converted_count = 0
    skipped_count = 0
    
    for task in data:
        print(f"attempting task {task['id']} ")
        # Extract image filename from the path
        if 'data' not in task or 'image' not in task['data']:
            print(f"Warning: Task {task.get('id', 'unknown')} has no image data, skipping")
            skipped_count += 1
            continue
        
        image_path = task['data']['image']
        # Extract filename from path like "/data/local-files/?d=images/0.20_119594044_YOLO_debug.jpg"
        if '?d=' in image_path:
            image_filename = image_path.split('?d=')[-1].split('/')[-1]
        else:
            image_filename = os.path.basename(image_path)
        
        # Create output filename (same name but .txt extension)
        txt_filename = os.path.splitext(image_filename)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)
        
        # Process annotations
        yolo_lines = []
        
        if 'annotations' not in task or len(task['annotations']) == 0:
            print(f"Warning: No annotations for {image_filename}, creating empty file")
            # Create empty file for images without annotations
            with open(txt_path, 'w') as f:
                pass
            converted_count += 1
            continue
        
        # Get the ground truth annotation (or first annotation if no ground truth)
        annotation = None
        for ann in task['annotations']:
            if ann.get('ground_truth', False):
                annotation = ann
                break
        if annotation is None:
            annotation = task['annotations'][0]
        
        if 'result' not in annotation:
            print(f"Warning: No results in annotation for {image_filename}, creating empty file")
            with open(txt_path, 'w') as f:
                pass
            converted_count += 1
            continue
        
        # Process each bounding box
        for result in annotation['result']:
            if result.get('type') != 'rectanglelabels':
                continue
            
            value = result.get('value', {})
            
            # Get label
            labels = value.get('rectanglelabels', [])
            if not labels:
                print(f"Warning: No label for bbox in {image_filename}, skipping bbox")
                continue
            
            label = labels[0]  # Take first label if multiple
            
            if label not in class_mapping:
                print(f"Warning: Label '{label}' not in class_mapping, skipping bbox")
                continue
            
            class_id = class_mapping[label]
            
            # Get bounding box coordinates (Label Studio uses percentages 0-100)
            x_percent = value.get('x') or 0  # top-left x
            y_percent = value.get('y') or 0  # top-left y
            width_percent = value.get('width') or 0
            height_percent = value.get('height') or 0
            
            # Convert to YOLO format (0-1 normalized, center coordinates)
            x_center = (x_percent + width_percent / 2) / 100.0
            y_center = (y_percent + height_percent / 2) / 100.0
            width_norm = width_percent / 100.0
            height_norm = height_percent / 100.0
            
            # Create YOLO format line
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
            yolo_lines.append(yolo_line)
        
        # Write to file
        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
            if yolo_lines:  # Add newline at end if file is not empty
                f.write('\n')
        
        converted_count += 1
        print(f"Converted task {task['id']} {image_filename} to {txt_filename} with {len(yolo_lines)} boxes")
        if converted_count % 100 == 0:
            print(f"Processed {converted_count} files...")
    
    print(f"\nConversion complete!")
    print(f"Converted: {converted_count} files")
    print(f"Skipped: {skipped_count} files")
    print(f"Output directory: {output_dir}")
    print(f"\nClass mapping used:")
    for label, idx in sorted(class_mapping.items(), key=lambda x: x[1]):
        print(f"  {idx}: {label}")
    
    # Save class mapping to classes.txt
    classes_file = os.path.join(output_dir, 'classes.txt')
    with open(classes_file, 'w') as f:
        for label, idx in sorted(class_mapping.items(), key=lambda x: x[1]):
            f.write(f"{label}\n")
    print(f"\nClass names saved to: {classes_file}")
    
    return class_mapping


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python labelstudio_to_yolo.py <input_json> <output_dir> [class_mapping_json]")
        print("\nExample:")
        print("  python labelstudio_to_yolo.py export.json ./labels/")
        print("\nOptional class mapping JSON format:")
        print('  {"Lily": 0, "Daisy": 1, "Rose": 2}')
        sys.exit(1)
    
    json_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    class_mapping = None
    if len(sys.argv) > 3:
        with open(sys.argv[3], 'r') as f:
            class_mapping = json.load(f)
    
    convert_labelstudio_to_yolo(json_file, output_dir, class_mapping)
