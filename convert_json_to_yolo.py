import json
import os
from pathlib import Path

# Configuration
JSON_LABELS_DIR = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/labeling_round2/done_integrate/osama200/json_labels"
OUTPUT_LABELS_DIR = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/labeling_round2/done_integrate/osama200/labels"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

# No longer needed - using group_id from JSON as class_id

def json_to_yolo(json_file):
    """Convert a single JSON file to YOLO format."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        image_width = data['imageWidth']
        image_height = data['imageHeight']
        shapes = data['shapes']
        
        yolo_lines = []
        
        for shape in shapes:
            if shape['shape_type'] != 'rectangle':
                print(f"Warning: Skipping non-rectangle shape in {json_file}")
                continue
            
            # Use group_id as class_id, and reset to start from 80
            class_id = shape.get('group_id')+80 
            if class_id is None:
                print(f"Warning: No group_id found in {json_file}")
                continue
            elif class_id in [80, 88]:  # Skip classes 80 and 88
                print(f"Info: Skipping class_id {class_id} in {json_file}")
                continue
            
            # Extract points [x1, y1] and [x2, y2]
            points = shape['points']
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            # Calculate center, width, height in pixels
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            # Normalize to 0-1 range
            norm_center_x = center_x / image_width
            norm_center_y = center_y / image_height
            norm_width = width / image_width
            norm_height = height / image_height
            
            # Create YOLO format line
            yolo_line = f"{class_id} {norm_center_x} {norm_center_y} {norm_width} {norm_height}"
            yolo_lines.append(yolo_line)
        
        # Save to YOLO format file
        if yolo_lines:
            base_name = Path(json_file).stem
            output_file = os.path.join(OUTPUT_LABELS_DIR, f"{base_name}.txt")
            
            with open(output_file, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            print(f"✓ Converted: {os.path.basename(json_file)} → {os.path.basename(output_file)} ({len(yolo_lines)} boxes)")
        else:
            print(f"⚠ Skipped: {os.path.basename(json_file)} (no valid shapes)")
        
        return True
    
    except Exception as e:
        print(f"✗ Error processing {json_file}: {str(e)}")
        return False

def main():
    if not os.path.exists(JSON_LABELS_DIR):
        print(f"Error: JSON labels directory not found: {JSON_LABELS_DIR}")
        return
    
    json_files = sorted(Path(JSON_LABELS_DIR).glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {JSON_LABELS_DIR}")
        return
    
    print(f"Found {len(json_files)} JSON files to convert\n")
    
    successful = 0
    for json_file in json_files:
        if json_to_yolo(json_file):
            successful += 1
    
    print(f"\n{'='*50}")
    print(f"Conversion complete: {successful}/{len(json_files)} files processed")
    print(f"Output saved to: {OUTPUT_LABELS_DIR}")

if __name__ == "__main__":
    main()
