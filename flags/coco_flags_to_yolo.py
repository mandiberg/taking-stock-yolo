#!/usr/bin/env python3

import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

# MODE 0: build generic flag training subset
# - Reads: flags/train_dataset.json
# - Images: flags/train_images/
# - Writes:
#   - flags/87_flag/images/
#   - flags/87_flag/labels/
MODE = 0

# Your reserved generic flag class in main model
GENERIC_FLAG_CLASS_ID = 87

GENERIC_FLAG_FOLDERNAME = "87_flag"
# .06 and .6 yields 5000 images
NON_PRIORITY_SAMPLE_FRACTION = 0.03
PRIORITY_SAMPLE_FRACTION = 0.3
RANDOM_SEED = 42

USE_ALL_FILES = [
    "China",
    "United_States_of_America",
    "India",
    "United_Kingdom",
    "Turkey",
    "Taiwan",
    "Thailand",
    "Japan",
    "Egypt",
    "France",
    "Israel",
    "Germany",
    "Ukraine",
    "Vietnam",
    "Brazil",
    "Australia",
]

# Mode-specific config
MODE_CONFIG = {
    0: {
        "json_path": Path("flags/train_dataset.json"),
        "images_dir": Path("flags/train_images"),
        "output_root": Path("flags") / GENERIC_FLAG_FOLDERNAME,
    }
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def coco_bbox_to_yolo(bbox, image_width: int, image_height: int):
    x, y, w, h = bbox

    x_center = (x + (w / 2.0)) / image_width
    y_center = (y + (h / 2.0)) / image_height
    w_norm = w / image_width
    h_norm = h / image_height

    return (
        clamp(x_center),
        clamp(y_center),
        clamp(w_norm),
        clamp(h_norm),
    )


def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_category_name(name: str) -> str:
    return name.strip().replace(" ", "_").lower()


def clear_directory(directory: Path) -> None:
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        return

    for child in directory.iterdir():
        if child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child)


def choose_image_subset(
    dataset: dict,
    non_priority_sample_fraction: float,
    priority_sample_fraction: float,
    random_seed: int,
):
    categories = dataset.get("categories", [])
    annotations = dataset.get("annotations", [])

    category_id_to_name = {category["id"]: category["name"] for category in categories}

    category_to_image_ids = defaultdict(set)
    for annotation in annotations:
        category_id = annotation["category_id"]
        image_id = annotation["image_id"]
        category_to_image_ids[category_id].add(image_id)

    prioritized_names_normalized = {
        normalize_category_name(country_name) for country_name in USE_ALL_FILES
    }

    selected_image_ids = set()
    stats = {
        "priority_categories_found": [],
        "priority_categories_missing": [],
        "priority_images_added": 0,
        "sampled_images_added": 0,
        "sampled_by_category": [],
    }

    rng = random.Random(random_seed)

    for category in categories:
        category_id = category["id"]
        category_name = category["name"]
        category_name_normalized = normalize_category_name(category_name)
        image_ids = sorted(category_to_image_ids.get(category_id, set()))

        if category_name_normalized in prioritized_names_normalized:
            if image_ids:
                sample_size = max(1, math.ceil(len(image_ids) * priority_sample_fraction))
                sample_size = min(sample_size, len(image_ids))
                sampled_ids = set(rng.sample(image_ids, sample_size))
                before = len(selected_image_ids)
                selected_image_ids.update(sampled_ids)
                added = len(selected_image_ids) - before
                stats["priority_images_added"] += added
                stats["priority_categories_found"].append(
                    (category_name, len(image_ids), sample_size, added)
                )
            else:
                stats["priority_categories_missing"].append(category_name)
            continue

        if not image_ids:
            continue

        sample_size = max(1, math.ceil(len(image_ids) * non_priority_sample_fraction))
        sample_size = min(sample_size, len(image_ids))
        sampled_ids = set(rng.sample(image_ids, sample_size))

        before = len(selected_image_ids)
        selected_image_ids.update(sampled_ids)
        added = len(selected_image_ids) - before
        stats["sampled_images_added"] += added
        stats["sampled_by_category"].append((category_name, len(image_ids), sample_size, added))

    priority_names_found = {
        normalize_category_name(item[0]) for item in stats["priority_categories_found"]
    }
    for country_name in USE_ALL_FILES:
        if normalize_category_name(country_name) not in priority_names_found:
            stats["priority_categories_missing"].append(country_name)

    return selected_image_ids, category_id_to_name, stats


def convert_mode_0() -> None:
    config = MODE_CONFIG[0]
    json_path = config["json_path"]
    images_dir = config["images_dir"]
    output_root = config["output_root"]

    output_images_dir = output_root / "images"
    output_labels_dir = output_root / "labels"

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    clear_directory(output_images_dir)
    clear_directory(output_labels_dir)

    dataset = load_coco(json_path)
    images = dataset.get("images", [])
    annotations = dataset.get("annotations", [])

    selected_image_ids, category_id_to_name, selection_stats = choose_image_subset(
        dataset,
        NON_PRIORITY_SAMPLE_FRACTION,
        PRIORITY_SAMPLE_FRACTION,
        RANDOM_SEED,
    )

    selected_images = [image for image in images if image["id"] in selected_image_ids]

    annotations_by_image_id = defaultdict(list)
    for annotation in annotations:
        if annotation["image_id"] in selected_image_ids:
            annotations_by_image_id[annotation["image_id"]].append(annotation)

    written_files = 0
    written_boxes = 0
    missing_images = 0
    copied_images = 0

    for image in selected_images:
        image_id = image["id"]
        image_width = image["width"]
        image_height = image["height"]
        file_name = image["file_name"]

        image_path = images_dir / file_name
        if not image_path.exists():
            missing_images += 1
            continue

        output_image_path = output_images_dir / file_name
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, output_image_path)
        copied_images += 1

        label_path = output_labels_dir / f"{Path(file_name).stem}.txt"

        yolo_lines = []
        for annotation in annotations_by_image_id.get(image_id, []):
            bbox = annotation.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            x_center, y_center, w_norm, h_norm = coco_bbox_to_yolo(
                bbox, image_width, image_height
            )
            yolo_lines.append(
                f"{GENERIC_FLAG_CLASS_ID} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            )

        with label_path.open("w", encoding="utf-8") as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines) + "\n")

        written_files += 1
        written_boxes += len(yolo_lines)

    print("MODE=0 complete")
    print(f"JSON: {json_path}")
    print(f"Images dir: {images_dir}")
    print(f"Output root: {output_root}")
    print(f"Output images: {output_images_dir}")
    print(f"Output labels: {output_labels_dir}")
    print(f"Images listed in JSON: {len(images)}")
    print(f"Selected images (subset): {len(selected_images)}")
    print(f"Annotations listed in JSON: {len(annotations)}")
    print(
        "Selected annotations: "
        f"{sum(len(annotations_by_image_id[image['id']]) for image in selected_images)}"
    )
    print(f"Copied images: {copied_images}")
    print(f"Label files written: {written_files}")
    print(f"Total YOLO boxes written: {written_boxes}")
    print(f"Missing image files skipped: {missing_images}")
    print(f"Priority countries requested: {len(USE_ALL_FILES)}")
    print(
        "Priority countries found in dataset: "
        f"{len(selection_stats['priority_categories_found'])}"
    )
    if selection_stats["priority_categories_missing"]:
        missing_unique = sorted(set(selection_stats["priority_categories_missing"]))
        print("Priority categories missing:", ", ".join(missing_unique))
    print(
        "Sampling rule for priority categories: "
        f"{int(PRIORITY_SAMPLE_FRACTION * 100)}%"
    )
    print(
        "Sampling rule for non-priority categories: "
        f"{int(NON_PRIORITY_SAMPLE_FRACTION * 100)}%"
    )
    print(f"Random seed: {RANDOM_SEED}")


if __name__ == "__main__":
    if MODE != 0:
        raise ValueError(f"Unsupported MODE={MODE}. Only MODE=0 is implemented now.")

    convert_mode_0()
