import os

from class_map_utils import resolve_class_name_to_id

FOLDER = "/Users/michael.mandiberg/Documents/YOLO_Training_Data/label_final steth"
CLASSES_FILE = os.path.join(FOLDER, 'classes.txt')
LABELS_FOLDER = os.path.join(FOLDER, 'labels')


def normalize_classes_line(raw_line: str) -> str:
    value = raw_line.strip()
    if not value:
        return value

    parts = value.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1]

    return value

def load_class_mappings(classes_file):
    class_id_to_YOLOid = {}
    with open(classes_file, 'r') as f:
        for local_id, line in enumerate(f):
            print(f"Line {local_id}: {line.strip()}")
            class_name = normalize_classes_line(line)
            global_id = resolve_class_name_to_id(class_name)
            if global_id is not None:
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