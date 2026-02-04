import os
import sys

'''
91_gun folder contains images and labels in separate folders.

Images folder has images in <code>jpeg</code> format. Labels folder has <code>txt</code> files where in each file, the first line contains the number of objects in the corresponding image and the next lines contain the co-ordinates of the box describing the object.

ex:

1
232 29 294 71

3
118 30 178 121
182 49 239 122
26 35 84 127

This script needs to format all the label files to YOLO format where each line contains:
<class_id> <x_center> <y_center> <width> <height>

'''

FOLDER_PATH = "/Users/michael.mandiberg/Documents/YOLO_Training_Data/sorted_images/91_gun"
IMAGES_FOLDER = os.path.join(FOLDER_PATH, "images")
LABELS_FOLDER = os.path.join(FOLDER_PATH, "labels")
OUTPUT_FOLDER = os.path.join(FOLDER_PATH, "labels_yolo")
CLASS_ID = 0  # 91_gun has only one class


def get_image_dimensions(image_path):
    """Get image width and height using PIL"""
    try:
        from PIL import Image
        img = Image.open(image_path)
        return img.width, img.height
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None, None


def convert_box_to_yolo(x1, y1, x2, y2, img_width, img_height):
    """
    Convert box coordinates from (x1, y1, x2, y2) to YOLO format.
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All values normalized to 0-1 range.
    """
    if img_width is None or img_height is None:
        return None
    
    # Calculate width and height
    width = x2 - x1
    height = y2 - y1
    
    # Calculate center
    x_center = x1 + width / 2
    y_center = y1 + height / 2
    
    # Normalize to 0-1
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return x_center_norm, y_center_norm, width_norm, height_norm


def process_labels():
    """Convert all label files from input format to YOLO format"""
    
    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    if not os.path.exists(LABELS_FOLDER):
        print(f"Labels folder not found: {LABELS_FOLDER}")
        return
    
    label_files = [f for f in os.listdir(LABELS_FOLDER) if f.endswith('.txt')]
    print(f"Found {len(label_files)} label files")
    
    converted_count = 0
    skipped_count = 0
    
    for label_file in label_files:
        label_path = os.path.join(LABELS_FOLDER, label_file)
        
        # Find corresponding image
        image_base = os.path.splitext(label_file)[0]
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            candidate = os.path.join(IMAGES_FOLDER, image_base + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break
        
        if image_path is None:
            print(f"Warning: No image found for {label_file}, skipping")
            skipped_count += 1
            continue
        
        # Get image dimensions
        img_width, img_height = get_image_dimensions(image_path)
        if img_width is None or img_height is None:
            print(f"Warning: Could not read dimensions for {image_path}, skipping")
            skipped_count += 1
            continue
        
        # Read and parse original label file
        try:
            with open(label_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            print(f"Error reading {label_path}: {e}")
            skipped_count += 1
            continue
        
        if len(lines) == 0:
            print(f"Warning: Empty label file {label_file}")
            skipped_count += 1
            continue
        
        # First line is count (we can verify but don't strictly need it)
        try:
            obj_count = int(lines[0])
        except ValueError:
            print(f"Warning: First line is not a number in {label_file}: {lines[0]}")
            skipped_count += 1
            continue
        
        # Parse boxes
        yolo_lines = []
        for i in range(1, len(lines)):
            parts = lines[i].split()
            if len(parts) != 4:
                print(f"Warning: Line {i} in {label_file} does not have 4 coordinates: {lines[i]}")
                continue
            
            try:
                x1, y1, x2, y2 = map(int, parts)
            except ValueError:
                print(f"Warning: Could not parse coordinates in {label_file} line {i}: {lines[i]}")
                continue
            
            result = convert_box_to_yolo(x1, y1, x2, y2, img_width, img_height)
            if result is None:
                continue
            
            x_center, y_center, width, height = result
            yolo_line = f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(yolo_line)
        
        # Write output file
        output_path = os.path.join(OUTPUT_FOLDER, label_file)
        try:
            with open(output_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
            converted_count += 1
            print(f"Converted: {label_file}")
        except Exception as e:
            print(f"Error writing {output_path}: {e}")
            skipped_count += 1
    
    print(f"\n✓ Conversion complete!")
    print(f"  Converted: {converted_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Output folder: {OUTPUT_FOLDER}")


if __name__ == '__main__':
    process_labels()
