import os
import re
import shutil
from pathlib import Path

def extract_numeric_id(filename):
    """
    Extracts a sequence of 5 or more digits to find the core numerical ID.
    Adjust the regex if your IDs are shorter than 5 digits.
    """
    filename = filename.split(".")[0]  # Remove file extension
    if filename.isdigit():
        return filename  # If the entire filename is digits, return it directly
    if "-" in filename: splitter = "-"
    elif "_" in filename: splitter = "_"
    else: splitter = None
    if splitter:
        parts = filename.split(splitter)
        if parts[0].isdigit() and parts[1].isdigit():
            return parts[1] 
        for part in parts:
            if part.isdigit():
                return part
        for part in parts:
            for c in part:
                if c.isdigit():
                    return part.replace("id", "")
                
    #         match = re.search(r'\d{5,}', part)
    #         if match:
    #             return match.group(0)

    # match = re.search(r'\d{5,}', filename)
    # return match.group(0) if match else None


def align_yolo_dataset(input_dir, output_dir):
    # Setup paths
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    img_in_dir = input_path / "images"
    lbl_in_dir = input_path / "labels"
    
    img_out_dir = output_path / "images"
    lbl_out_dir = output_path / "labels"
    unmatched_dir = output_path / "unmatched"
    
    # Create output directories
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)
    unmatched_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionaries to hold map of {numeric_id: full_file_path}
    image_map = {}
    label_map = {}
    unmatched_image_map = {}
    unmatched_label_map = {}
    
    # Scan images
    if img_in_dir.exists():
        for img_file in img_in_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                num_id = extract_numeric_id(img_file.stem)
                print(f"Processing image: {img_file.name}, extracted ID: {num_id}")
                if num_id:
                    image_map[num_id] = img_file
                else:
                    print(f"Skipping image (No numeric ID found): {img_file.name}")
                    shutil.copy(img_file, unmatched_dir / img_file.name)
                    
    # Scan labels
    if lbl_in_dir.exists():
        for lbl_file in lbl_in_dir.iterdir():
            if lbl_file.is_file() and lbl_file.suffix.lower() == '.txt':
                num_id = extract_numeric_id(lbl_file.stem)
                print(f"Processing label: {lbl_file.name}, extracted ID: {num_id}")
                if num_id:
                    label_map[num_id] = lbl_file
                else:
                    print(f"Skipping label (No numeric ID found): {lbl_file.name}")
                    shutil.copy(lbl_file, unmatched_dir / lbl_file.name)
    # print("image_map:", image_map)
    # print("label_map:", label_map)
    # Find common IDs and process
    all_ids = set(image_map.keys()).union(set(label_map.keys()))
    
    matched_count = 0
    unmatched_count = 0
    
    for num_id in all_ids:
        has_image = num_id in image_map
        has_label = num_id in label_map
        
        if has_image and has_label:
            # Reconstruct clean filenames based strictly on the identifier
            img_file = image_map[num_id]
            lbl_file = label_map[num_id]
            
            # Keep original image extension (.jpg, .png, etc.)
            new_img_name = f"{num_id}{img_file.suffix.lower()}"
            new_lbl_name = f"{num_id}.txt"
            
            # Copy to clean structure
            shutil.copy(img_file, img_out_dir / new_img_name)
            shutil.copy(lbl_file, lbl_out_dir / new_lbl_name)
            matched_count += 1
        else:
            # store unmatched files in     unmatched_image_map = {}
    # unmatched_label_map = {}
            unmatched_image_map[num_id] = image_map.get(num_id)
            unmatched_label_map[num_id] = label_map.get(num_id)




            # print(f"Unmatched files for ID {num_id}: "
            #       f"{'Image found' if has_image else 'No image'}, "
            #       f"{'Label found' if has_label else 'No label'}")
    
    for num_id, img_file in unmatched_image_map.items():
        if img_file:
            # check to see if img_file.name is in any lbl_file.name
            for lbl_file in label_map.values():
                if img_file.stem in lbl_file.stem:
                    print(f"Found matching label for unmatched image {img_file.name}: {lbl_file.name}")
                    # Copy to clean structure
                    new_img_name = f"{num_id}{img_file.suffix.lower()}"
                    new_lbl_name = f"{num_id}.txt"
                    shutil.copy(img_file, img_out_dir / new_img_name)
                    shutil.copy(lbl_file, lbl_out_dir / new_lbl_name)
                    matched_count += 1
                    # remove from unmatched maps since they are now matched
                    unmatched_image_map[num_id] = None
                    unmatched_label_map[extract_numeric_id(lbl_file.stem)] = None
                    break
    for num_id, lbl_file in unmatched_label_map.items():
        if lbl_file:
            # check to see if lbl_file.name is in any img_file.name
            for img_file in image_map.values():
                if lbl_file.stem in img_file.stem:
                    repaired = True
                else:
                    # chop off the first 15 characters, and last 15 characters, and see if that is in the img_file.name
                    lbl_chunk = lbl_file.stem[15:]
                    lbl_chunk = lbl_chunk[:15]
                    if lbl_chunk in img_file.stem:
                        repaired = True
                    else:
                        repaired = False
                if repaired:
                    print(f"Found matching image for unmatched label {lbl_file.name}: {img_file.name}")
                    # Copy to clean structure
                    new_img_name = f"{num_id}{img_file.suffix.lower()}"
                    new_lbl_name = f"{num_id}.txt"
                    shutil.copy(img_file, img_out_dir / new_img_name)
                    shutil.copy(lbl_file, lbl_out_dir / new_lbl_name)
                    matched_count += 1
                    # remove from unmatched maps since they are now matched
                    unmatched_image_map[extract_numeric_id(img_file.stem)] = None
                    unmatched_label_map[num_id] = None
                    break
                # else:
                #     # Move orphans to the inspection folder
                #     if image_map[num_id]:
                #         shutil.copy(image_map[num_id], unmatched_dir / image_map[num_id].name)
                #     if label_map[num_id]:
                #         shutil.copy(label_map[num_id], unmatched_dir / label_map[num_id].name)
                #     unmatched_count += 1
    for num_id, img_file in unmatched_image_map.items():
        print(f"Unmatched image for ID {num_id}: {img_file.name if img_file else 'No image'}")
        if img_file:
            shutil.copy(img_file, unmatched_dir / img_file.name)
            unmatched_count += 1
    for num_id, lbl_file in unmatched_label_map.items():
        print(f"Unmatched label for ID {num_id}: {lbl_file.name if lbl_file else 'No label'}")
        if img_file:
            shutil.copy(lbl_file, unmatched_dir / lbl_file.name)
            unmatched_count += 1
            

    print("\n=== Process Complete ===")
    print(f"Successfully paired and renamed: {matched_count} pairs")
    print(f"Unmatched files isolated: {unmatched_count}")
    print(f"Clean dataset saved to: {output_path.resolve()}")

# --- Configuration ---
if __name__ == "__main__":
    # Replace these with your actual folder paths
    '''
    140_Boxing_gloves
141_soccerball
142_basketball
143_football
144_volleyball
145_baseball
146_tennisball
147_weightball
148_discoball
149_beachball
150_yogaball
151_yogamat
'''
    INPUT_DATASET_FOLDER = "/Users/michaelmandiberg/Documents/YOLO_Training_Data/sorted_images_orig/151_yogamat" 
    OUTPUT_CLEAN_FOLDER = "/Users/michaelmandiberg/Documents/YOLO_Training_Data/sorted_images/151_yogamat"
    
    align_yolo_dataset(INPUT_DATASET_FOLDER, OUTPUT_CLEAN_FOLDER)
