import os


name_to_id = {
    'Sign': 80,
    'Gift': 81,
    'Money': 82,
    'Bag': 83,
    'Valentine': 84,
    'Salad': 85,
    'Dumbbell': 86,
    'Flag': 87,
    'Groceries': 88,
    'Mask': 89,
    'Stethoscope': 90,
    'Gun': 91,
    'Headphones': 92,
    'Clipboard': 93,
    'Piggybank': 94,
    'Creditcard': 95,
    'Bitcoin': 96,
    'Rose': 97,
    'Lily': 98,
    'Iris': 99,
    'Tulip': 100,
    'Lisianthus': 101,
    'Orchid': 102,
    'Peony': 103
}

FOLDER = "/Users/michael.mandiberg/Documents/YOLO_Training_Data/90_stethoscope"
CLASSES_FILE = os.path.join(FOLDER, 'classes.txt')
LABELS_FOLDER = os.path.join(FOLDER, 'labels')

def load_class_mappings(classes_file):
    class_id_to_YOLOid = {}
    with open(classes_file, 'r') as f:
        for local_id, line in enumerate(f):
            print(f"Line {local_id}: {line.strip()}")
            class_name = line.strip().lower()
            if class_name in {k.lower(): v for k, v in name_to_id.items()}:
                global_id = name_to_id[next(k for k in name_to_id if k.lower() == class_name)]
                class_id_to_YOLOid[local_id] = global_id
    print("Loaded class ID to YOLO ID mapping:", class_id_to_YOLOid)
    return class_id_to_YOLOid

if __name__ == "__main__":
    class_id_to_YOLOid = load_class_mappings(CLASSES_FILE)
    print("Class ID to YOLO ID mapping:", class_id_to_YOLOid)

    for label_file in os.listdir(LABELS_FOLDER):
        if not label_file.endswith('.txt'):
            continue
        label_path = os.path.join(LABELS_FOLDER, label_file)

        with open(label_path, 'r') as f:
            lines = f.readlines()

        converted_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                old_class_id = int(parts[0])
                # Convert to YOLO ID
                yolo_id = class_id_to_YOLOid.get(old_class_id, old_class_id)
                parts[0] = str(yolo_id)
                parts[3] = '0.99' if float(parts[3]) >= 1 else parts[3]  # width
                parts[4] = '0.99' if float(parts[4]) >= 1 else parts[4]  # height
                converted_lines.append(' '.join(parts) + '\n')
                print(f"Converted {old_class_id} to {yolo_id} in line: {line.strip()}")

        with open(label_path, 'w') as f:
            f.writelines(converted_lines)

        print(f"Converted  saved to: {label_path}")