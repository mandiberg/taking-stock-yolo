import json

# Load original COCO dataset
with open('train_dataset.json', 'r') as f:
    coco_data = json.load(f)

# Find images
images_by_name = {img['file_name']: img for img in coco_data['images']}

# Create category mapping
cat_by_id = {c['id']: c['name'] for c in coco_data['categories']}

# Look for the specific images
for img_name in ['Benin_007.png', 'San_Marino_006.png']:
    if img_name in images_by_name:
        img = images_by_name[img_name]
        img_id = img['id']
        print(f"\n=== {img_name} ===")
        print(f"Image ID: {img_id}")
        print(f"Dimensions: {img['width']}x{img['height']}")
        
        # Find all annotations for this image
        annots = [a for a in coco_data['annotations'] if a['image_id'] == img_id]
        print(f"Number of annotations: {len(annots)}")
        
        for i, annot in enumerate(annots):
            cat_id = annot['category_id']
            cat_name = cat_by_id.get(cat_id, 'UNKNOWN')
            bbox = annot['bbox']
            print(f"  [{i}] Category {cat_id} ({cat_name}), bbox: {bbox}")
    else:
        print(f"\n{img_name} not found in dataset")

# Now check the MODE 1 labels
print("\n\n=== MODE 1 CONVERTED LABELS ===")
for img_name in ['Benin_007.png', 'San_Marino_006.png']:
    label_file = f"multi_flag/labels/{img_name.replace('.png', '.txt')}"
    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        print(f"\n{img_name}:")
        print(f"  Number of label lines: {len(lines)}")
        for i, line in enumerate(lines):
            print(f"  [{i}] {line.strip()}")
    except FileNotFoundError:
        print(f"\n{img_name}: LABEL FILE NOT FOUND")
