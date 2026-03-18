#!/usr/bin/env python3

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze category distribution from a COCO-style flag dataset JSON."
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("flags/train_dataset.json"),
        help="Path to COCO JSON file (default: flags/test_dataset.json)",
    )
    parser.add_argument(
        "--sort-by",
        choices=["annotations", "images", "name"],
        default="annotations",
        help="Sort output by annotation count, unique image count, or name.",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending instead of descending.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Show only top N categories (0 means show all).",
    )
    return parser.parse_args()


def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_counts(dataset: dict):
    categories = dataset.get("categories", [])
    annotations = dataset.get("annotations", [])

    category_id_to_name = {category["id"]: category["name"] for category in categories}

    annotation_counts = Counter()
    category_to_image_ids = defaultdict(set)

    for annotation in annotations:
        category_id = annotation["category_id"]
        image_id = annotation["image_id"]
        annotation_counts[category_id] += 1
        category_to_image_ids[category_id].add(image_id)

    rows = []
    for category_id, category_name in category_id_to_name.items():
        rows.append(
            {
                "category_id": category_id,
                "category_name": category_name,
                "annotation_count": annotation_counts.get(category_id, 0),
                "image_count": len(category_to_image_ids.get(category_id, set())),
            }
        )

    return rows, len(annotations), len(categories)


def sort_rows(rows, sort_by: str, ascending: bool):
    if sort_by == "annotations":
        key = lambda row: (row["annotation_count"], row["category_name"])
    elif sort_by == "images":
        key = lambda row: (row["image_count"], row["category_name"])
    else:
        key = lambda row: row["category_name"].lower()

    return sorted(rows, key=key, reverse=not ascending)


def print_table(rows, total_annotations: int, total_categories: int):
    print(f"Total categories: {total_categories}")
    print(f"Total annotations: {total_annotations}")
    print("-" * 80)
    print(f"{'category_id':>11}  {'annotation_count':>16}  {'image_count':>11}  category_name")
    print("-" * 80)

    for row in rows:
        print(
            f"{row['category_id']:>11}  {row['annotation_count']:>16}  {row['image_count']:>11}  {row['category_name']}"
        )


def main() -> None:
    args = parse_args()

    if not args.json.exists():
        raise FileNotFoundError(f"JSON file not found: {args.json}")

    dataset = load_coco(args.json)
    rows, total_annotations, total_categories = compute_counts(dataset)
    rows = sort_rows(rows, args.sort_by, args.ascending)

    if args.top > 0:
        rows = rows[: args.top]

    print_table(rows, total_annotations, total_categories)


if __name__ == "__main__":
    main()
