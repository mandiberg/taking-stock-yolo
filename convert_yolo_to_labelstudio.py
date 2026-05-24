import os
import json
import re
import cv2

from class_map_utils import get_id_to_name

# 📌 CONFIG — adjust these
YOLO_ROOT = "/Users/michaelmandiberg/Documents/yolo/reprocess/"
# YOLO_ROOT = "/Volumes/LaCie/test_output/label_studio_ready"
WALK_FOLDERS = False

# DATASET_FOLDER = "sort/relabel_these"
DATASET_FOLDER = "relabel_these93_midzone"
ONLY_MULTIPLE_BOXES = False
SUPPRESS_NON_ALIGNED_ANNOTATIONS = False


def parse_batch_class_id_from_dataset_folder(dataset_folder):
    match = re.search(r"\d+", dataset_folder)
    if match is None:
        return None
    return int(match.group(0))


def is_hidden_name(name):
    return name.startswith(".")


def get_suppression_config(dataset_folder):
    batch_class_id = parse_batch_class_id_from_dataset_folder(dataset_folder)
    suppression_enabled = SUPPRESS_NON_ALIGNED_ANNOTATIONS and batch_class_id is not None
    if SUPPRESS_NON_ALIGNED_ANNOTATIONS and batch_class_id is None:
        print(
            "SUPPRESS_NON_ALIGNED_ANNOTATIONS=True but no integer was found in "
            f"dataset folder '{dataset_folder}'. Proceeding without suppression."
        )
    if suppression_enabled:
        print(
            "Suppressing non-aligned annotations using batch class ID "
            f"{batch_class_id} parsed from dataset folder '{dataset_folder}'."
        )
    return suppression_enabled, batch_class_id

# Canonical classes are loaded from config/custom_class_map.json
CLASSES = get_id_to_name(min_id=80, max_id=128)

# CLASSES = {
#     0: "Afghanistan",
#     1: "Albania",
#     2: "Algeria",
#     3: "American Samoa",
#     4: "Andorra",
#     5: "Angola",
#     6: "Anguilla",
#     7: "Antigua and Barbuda",
#     8: "Argentina",
#     9: "Armenia",
#     10: "Aruba",
#     11: "Australia",
#     12: "Austria",
#     13: "Azerbaijan",
#     14: "Bahamas",
#     15: "Bahrain",
#     16: "Bangladesh",
#     17: "Barbados",
#     18: "Belarus",
#     19: "Belgium",
#     20: "Belize",
#     21: "Benin",
#     22: "Bermuda",
#     23: "Bhutan",
#     24: "Bolivia",
#     25: "Bosnia",
#     26: "Botswana",
#     27: "Brazil",
#     28: "British Virgin Islands",
#     29: "Brunei",
#     30: "Bulgaria",
#     31: "Burkina Faso",
#     32: "Burundi",
#     33: "Cambodia",
#     34: "Cameroon",
#     35: "Canada",
#     36: "Cape Verde",
#     37: "Cayman Islands",
#     38: "Central African Republic",
#     39: "Chad",
#     40: "Chile",
#     41: "China",
#     42: "Christmas Island",
#     43: "Colombia",
#     44: "Comoros",
#     45: "Cook Islands",
#     46: "Costa Rica",
#     47: "Croatia",
#     48: "Cuba",
#     49: "Cyprus",
#     50: "Czech Republic",
#     51: "Democratic Republic of the Congo",
#     52: "Denmark",
#     53: "Djibouti",
#     54: "Dominica",
#     55: "Dominican Republic",
#     56: "Ecuador",
#     57: "Egypt",
#     58: "El Salvador",
#     59: "Equatorial Guinea",
#     60: "Eritrea",
#     61: "Estonia",
#     62: "Ethiopia",
#     63: "Falkland Islands",
#     64: "Faroe Islands",
#     65: "Fiji",
#     66: "Finland",
#     67: "France",
#     68: "French Polynesia",
#     69: "Gabon",
#     70: "Gambia",
#     71: "Georgia",
#     72: "Germany",
#     73: "Ghana",
#     74: "Gibraltar",
#     75: "Greece",
#     76: "Greenland",
#     77: "Grenada",
#     78: "Guam",
#     79: "Guatemala",
#     80: "Guinea",
#     81: "Guinea Bissau",
#     82: "Guyana",
#     83: "Haiti",
#     84: "Honduras",
#     85: "Hong Kong",
#     86: "Hungary",
#     87: "Iceland",
#     88: "India",
#     89: "Indonesia",
#     90: "Iran",
#     91: "Iraq",
#     92: "Ireland",
#     93: "Israel",
#     94: "Italy",
#     95: "Ivory Coast",
#     96: "Jamaica",
#     97: "Japan",
#     98: "Jordan",
#     99: "Kazakhstan",
#     100: "Kenya",
#     101: "Kiribati",
#     102: "Kuwait",
#     103: "Kyrgyzstan",
#     104: "Laos",
#     105: "Latvia",
#     106: "Lebanon",
#     107: "Lesotho",
#     108: "Liberia",
#     109: "Libya",
#     110: "Liechtenstein",
#     111: "Lithuania",
#     112: "Luxembourg",
#     113: "Macao",
#     114: "Macedonia",
#     115: "Madagascar",
#     116: "Malawi",
#     117: "Malaysia",
#     118: "Maldives",
#     119: "Mali",
#     120: "Malta",
#     121: "Marshall Islands",
#     122: "Mauritania",
#     123: "Mauritius",
#     124: "Mexico",
#     125: "Micronesia",
#     126: "Moldova",
#     127: "Monaco",
#     128: "Mongolia",
#     129: "Montenegro",
#     130: "Montserrat",
#     131: "Morocco",
#     132: "Mozambique",
#     133: "Myanmar",
#     134: "Namibia",
#     135: "Nauru",
#     136: "Nepal",
#     137: "Netherlands",
#     138: "Netherlands Antilles",
#     139: "New Zealand",
#     140: "Nicaragua",
#     141: "Niger",
#     142: "Nigeria",
#     143: "Niue",
#     144: "Norfolk Island",
#     145: "North Korea",
#     146: "Norway",
#     147: "Oman",
#     148: "Others",
#     149: "Pakistan",
#     150: "Palau",
#     151: "Panama",
#     152: "Papua New Guinea",
#     153: "Paraguay",
#     154: "Peru",
#     155: "Philippines",
#     156: "Pitcairn Islands",
#     157: "Poland",
#     158: "Portugal",
#     159: "Puerto Rico",
#     160: "Qatar",
#     161: "Republic of the Congo",
#     162: "Romania",
#     163: "Russian Federation",
#     164: "Rwanda",
#     165: "Saint Kitts and Nevis",
#     166: "Saint Lucia",
#     167: "Saint Pierre",
#     168: "Saint Vicent and the Grenadines",
#     169: "Samoa",
#     170: "San Marino",
#     171: "Sao Tome and Principe",
#     172: "Saudi Arabia",
#     173: "Senegal",
#     174: "Serbia",
#     175: "Serbia and Montenegro",
#     176: "Seychelles",
#     177: "Sierra Leone",
#     178: "Singapore",
#     179: "Slovakia",
#     180: "Slovenia",
#     181: "Soloman Islands",
#     182: "Somalia",
#     183: "South Africa",
#     184: "South Georgia",
#     185: "South Korea",
#     186: "South Sudan",
#     187: "Spain",
#     188: "Sri Lanka",
#     189: "Sudan",
#     190: "Suriname",
#     191: "Swaziland",
#     192: "Sweden",
#     193: "Switzerland",
#     194: "Syria",
#     195: "Taiwan",
#     196: "Tajikistan",
#     197: "Tanzania",
#     198: "Thailand",
#     199: "Tibet",
#     200: "Timor Leste",
#     201: "Togo",
#     202: "Tonga",
#     203: "Trinidad and Tobago",
#     204: "Tunisia",
#     205: "Turkey",
#     206: "Turkmenistan",
#     207: "Turks and Caicos Islands",
#     208: "Tuvalu",
#     209: "UAE",
#     210: "Uganda",
#     211: "Ukraine",
#     212: "United Kingdom",
#     213: "United States of America",
#     214: "Uruguay",
#     215: "US Virgin Islands",
#     216: "Uzbekistan",
#     217: "Vanuatu",
#     218: "Vatican City",
#     219: "Venezuela",
#     220: "Vietnam",
#     221: "Wallis and Futuna",
#     222: "Yemen",
#     223: "Zambia",
#     224: "Zimbabwe",
#     225: "European Union",
#     226: "Pride",
#     227: "United Nations",
#     228: "Checkered Flag",
# }

def process_dataset(dataset_folder):
    images_dir = os.path.join(YOLO_ROOT, dataset_folder, "images")
    labels_dir = os.path.join(YOLO_ROOT, dataset_folder, "labels")
    output_json = os.path.join(YOLO_ROOT, dataset_folder, "labelstudio_tasks.json")
    suppression_enabled, batch_class_id = get_suppression_config(dataset_folder)

    print(f"\nProcessing dataset folder: {dataset_folder}")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")

    tasks = []
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        print(f"[WARN] Missing images/ or labels/ for dataset '{dataset_folder}'. Writing empty JSON.")
        with open(output_json, "w") as f:
            json.dump(tasks, f, indent=2)
        print(f"Done! Label Studio tasks written to:\n  {output_json}")
        return 0

    for img_name in sorted(os.listdir(images_dir)):
        if is_hidden_name(img_name):
            continue
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(images_dir, img_name)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image {img_name}, skipping.")
            continue

        base = os.path.splitext(img_name)[0]
        label_txt = os.path.join(labels_dir, base + ".txt")
        if not os.path.exists(label_txt):
            print(f"No label file found for {img_name}, skipping.")
            continue

        results = []

        with open(label_txt, "r") as f:
            lines = f.readlines()

        valid_box_count = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                line_class_id = int(parts[0])
                if suppression_enabled and line_class_id != batch_class_id:
                    continue
                valid_box_count += 1

        if ONLY_MULTIPLE_BOXES and valid_box_count < 2:
            print(f"Skipping {img_name}: has {valid_box_count} box(es), need >= 2.")
            continue

        with open(label_txt, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"Skipping malformed line in {label_txt}: {line.strip()}")
                    continue

                class_id, xc, yc, bw, bh = parts
                class_id = int(class_id)
                if suppression_enabled and class_id != batch_class_id:
                    continue
                xc, yc, bw, bh = map(float, [xc, yc, bw, bh])

                left = (xc - bw / 2) * 100
                top = (yc - bh / 2) * 100
                width = bw * 100
                height = bh * 100

                print(f"Converting box for class ID {class_id} in {img_name} with {len(CLASSES)} classes.")
                label_name = CLASSES.get(class_id, f"class{class_id}")
                print(f" - Found box: {label_name} at ({left:.2f}%, {top:.2f}%, {width:.2f}%, {height:.2f}%)")

                result = {
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": left,
                        "y": top,
                        "width": width,
                        "height": height,
                        "rectanglelabels": [label_name],
                    },
                }
                results.append(result)

        if not results:
            print(f"No boxes for {img_name}, skipping.")
            continue

        image_folder = "/data/local-files/?d=images/"
        img_data_path = os.path.join(image_folder, img_name)
        task = {
            "data": {
                "image": img_data_path,
            },
            "annotations": [
                {
                    "result": results,
                }
            ],
        }

        tasks.append(task)
        print(f"Converted {img_name} with {len(results)} boxes")

    with open(output_json, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"Done! Label Studio tasks written to:\n  {output_json}")
    return len(tasks)


def discover_dataset_folders(yolo_root):
    dataset_folders = []
    for name in sorted(os.listdir(yolo_root)):
        if is_hidden_name(name):
            continue
        dataset_path = os.path.join(yolo_root, name)
        if not os.path.isdir(dataset_path):
            continue
        images_dir = os.path.join(dataset_path, "images")
        labels_dir = os.path.join(dataset_path, "labels")
        if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
            dataset_folders.append(name)
        else:
            print(f"Skipping {name}: missing images/ or labels/")
    return dataset_folders


def main():
    print(f"Processing YOLO root at:\n  {YOLO_ROOT}")
    if not os.path.isdir(YOLO_ROOT):
        print(f"YOLO_ROOT not found: {YOLO_ROOT}")
        return

    if WALK_FOLDERS:
        dataset_folders = discover_dataset_folders(YOLO_ROOT)
        if not dataset_folders:
            print("No valid dataset folders found.")
            return

        total_tasks = 0
        for dataset_folder in dataset_folders:
            total_tasks += process_dataset(dataset_folder)

        print("\nSummary:")
        print(f"Datasets processed: {len(dataset_folders)}")
        print(f"Total tasks written: {total_tasks}")
    else:
        process_dataset(DATASET_FOLDER)


if __name__ == "__main__":
    main()
