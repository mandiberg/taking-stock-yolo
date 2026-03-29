"""
Strict detective script: Find multi-instance images with uniform (incomplete) labeling.

Criteria for "suspicious" image:
- Has 2+ bounding boxes (annotations)
- ALL annotations have the SAME category_id
- => Suggests the labeling team only annotated one type of flag despite multiple being visible

Moves suspicious images+labels to a quarantine folder for manual review.
"""

import json
from pathlib import Path
import shutil
from collections import defaultdict

# Paths
flags_dir = Path(".")
json_path = flags_dir / "train_dataset.json"
multi_flag_dir = flags_dir / "multi_flag"
output_dir = multi_flag_dir / "untitled folder"
output_images_dir = output_dir / "images"
output_labels_dir = output_dir / "labels"

print("=== SUSPICIOUS MULTI-FLAG IMAGE DETECTOR ===\n")

# Create output directories
output_images_dir.mkdir(parents=True, exist_ok=True)
output_labels_dir.mkdir(parents=True, exist_ok=True)

# Load COCO dataset
print(f"Loading {json_path}...")
with open(json_path, 'r') as f:
    coco_data = json.load(f)

images = coco_data['images']
annotations = coco_data['annotations']
categories = {c['id']: c['name'] for c in coco_data['categories']}

print(f"Loaded {len(images)} images, {len(annotations)} annotations, {len(categories)} categories\n")

# Build image metadata
images_by_id = {img['id']: img for img in images}

# Group annotations by image_id
annots_by_image = defaultdict(list)
for annot in annotations:
    annots_by_image[annot['image_id']].append(annot)

# Find suspicious images (2+ annotations, all same category)
suspicious = []
for img_id, annots in annots_by_image.items():
    if len(annots) >= 2:
        category_ids = [a['category_id'] for a in annots]
        # All the same?
        if len(set(category_ids)) == 1:
            img = images_by_id[img_id]
            suspicious.append({
                'image_id': img_id,
                'file_name': img['file_name'],
                'num_boxes': len(annots),
                'category_id': category_ids[0],
                'category_name': categories.get(category_ids[0], 'UNKNOWN'),
                'width': img['width'],
                'height': img['height'],
                'bboxes': [a['bbox'] for a in annots]
            })

print(f"Found {len(suspicious)} suspicious images (2+ boxes, all same category)\n")

# Move suspicious images and labels
moved_count = 0
not_found_count = 0

for suspect in suspicious:
    file_name = suspect['file_name']
    stem = Path(file_name).stem
    
    # Source files
    img_src = multi_flag_dir / "images" / file_name
    label_src = multi_flag_dir / "labels" / f"{stem}.txt"
    
    # Destination files
    img_dst = output_images_dir / file_name
    label_dst = output_labels_dir / f"{stem}.txt"
    
    # Move image
    if img_src.exists():
        shutil.move(str(img_src), str(img_dst))
    else:
        not_found_count += 1
        print(f"  WARNING: Image not found: {img_src}")
    
    # Move label
    if label_src.exists():
        shutil.move(str(label_src), str(label_dst))
    else:
        print(f"  WARNING: Label not found: {label_src}")
    
    moved_count += 1

print(f"Moved {moved_count} suspicious image+label pairs to {output_dir}")
if not_found_count > 0:
    print(f"  ({not_found_count} images were not found in multi_flag/images)")

# Write detailed report
report_path = multi_flag_dir / "suspicious_images_report.json"
with open(report_path, 'w') as f:
    json.dump({
        'num_suspicious': len(suspicious),
        'detection_criteria': {
            'min_annotations': 2,
            'requirement': 'all annotations must have same category_id'
        },
        'images': suspicious
    }, f, indent=2)

print(f"\nDetailed report written to: {report_path}")

# Create a summary report for quick review
summary_path = multi_flag_dir / "suspicious_images_summary.txt"
with open(summary_path, 'w') as f:
    f.write("SUSPICIOUS MULTI-FLAG IMAGES - MANUAL REVIEW NEEDED\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Total suspicious images: {len(suspicious)}\n")
    f.write(f"Criteria: 2+ bounding boxes with all same category_id\n")
    f.write(f"Interpretation: Likely incomplete labeling (only one flag type annotated)\n\n")
    f.write("Images:\n")
    f.write("-" * 70 + "\n")
    
    for i, suspect in enumerate(suspicious, 1):
        f.write(f"{i}. {suspect['file_name']}\n")
        f.write(f"   Image ID: {suspect['image_id']}\n")
        f.write(f"   Dimensions: {suspect['width']}x{suspect['height']}\n")
        f.write(f"   Annotation count: {suspect['num_boxes']}\n")
        f.write(f"   All labeled as: {suspect['category_name']} (category {suspect['category_id']})\n")
        f.write(f"   Bboxes: {suspect['bboxes']}\n")
        f.write("\n")

print(f"Summary report written to: {summary_path}\n")

# Print quick summary
print("TOP 10 SUSPICIOUS IMAGES:")
print("=" * 70)
for i, suspect in enumerate(suspicious[:10], 1):
    print(f"{i}. {suspect['file_name']:40s} | {suspect['num_boxes']} boxes, all {suspect['category_name']}")

if len(suspicious) > 10:
    print(f"... and {len(suspicious) - 10} more\n")

print("\nNext steps:")
print("1. Review images in:", output_dir)
print("2. Decide which to keep/discard")
print("3. If keeping any, move them back to multi_flag/images and multi_flag/labels")
print("4. Update convert_mode_1() to exclude problematic images if needed")
