import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image as PilImage
from transformers import pipeline

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

'''
python bootstrap_open_vocab_labels.py \
  --images-dir /Volumes/LaCie/segment_images_89_mask/images_adobe/A/AA \
  --output-dir /Volumes/LaCie/segment_images_89_mask/bootstrap/images_adobe/A/AA \
  --classes "lipstick,Cosmetics Brush/Eyeliner Pencil,Towel"

'''
def parse_classes(raw: str) -> List[str]:
    classes = [item.strip() for item in raw.split(",") if item.strip()]
    if not classes:
        raise ValueError("No classes provided. Use --classes 'class1,class2,...'")
    return classes


def load_prompt_mapping(classes: List[str], prompts_json_path: str | None) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    class_to_prompts: Dict[str, List[str]] = {name: [name] for name in classes}

    if prompts_json_path:
        with open(prompts_json_path, "r", encoding="utf-8") as f:
            user_mapping = json.load(f)
        for class_name, prompts in user_mapping.items():
            if class_name not in class_to_prompts:
                continue
            if isinstance(prompts, list) and prompts:
                class_to_prompts[class_name] = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]

    prompt_to_class: Dict[str, str] = {}
    for class_name, prompts in class_to_prompts.items():
        for prompt in prompts:
            prompt_to_class[prompt] = class_name

    return class_to_prompts, prompt_to_class


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def xyxy_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float, width: int, height: int) -> Tuple[float, float, float, float]:
    left = clip(xmin, 0.0, float(width))
    right = clip(xmax, 0.0, float(width))
    top = clip(ymin, 0.0, float(height))
    bottom = clip(ymax, 0.0, float(height))

    if right <= left or bottom <= top:
        return 0.0, 0.0, 0.0, 0.0

    x_center = ((left + right) / 2.0) / width
    y_center = ((top + bottom) / 2.0) / height
    box_width = (right - left) / width
    box_height = (bottom - top) / height

    return x_center, y_center, box_width, box_height


def iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def classwise_nms(detections: List[dict], nms_iou: float) -> List[dict]:
    kept: List[dict] = []
    by_class: Dict[int, List[dict]] = {}

    for detection in detections:
        by_class.setdefault(detection["class_id"], []).append(detection)

    for _, items in by_class.items():
        sorted_items = sorted(items, key=lambda item: item["score"], reverse=True)
        selected: List[dict] = []

        for candidate in sorted_items:
            candidate_box = (
                candidate["xmin"],
                candidate["ymin"],
                candidate["xmax"],
                candidate["ymax"],
            )
            suppressed = False
            for picked in selected:
                picked_box = (
                    picked["xmin"],
                    picked["ymin"],
                    picked["xmax"],
                    picked["ymax"],
                )
                if iou(candidate_box, picked_box) >= nms_iou:
                    suppressed = True
                    break
            if not suppressed:
                selected.append(candidate)

        kept.extend(selected)

    return kept


def queue_for_score(score: float, high_threshold: float, low_threshold: float) -> str:
    if score >= high_threshold:
        return "accept"
    if score >= low_threshold:
        return "review"
    return "discard"


def detect_image(detector, pil_image: PilImage.Image, prompt_list: List[str], min_conf: float) -> List[dict]:
    predictions = detector(pil_image, candidate_labels=prompt_list)

    filtered = []
    for pred in predictions:
        score = float(pred.get("score", 0.0))
        if score < min_conf:
            continue
        box = pred.get("box", {})
        filtered.append(
            {
                "prompt": pred.get("label", ""),
                "score": score,
                "xmin": float(box.get("xmin", 0.0)),
                "ymin": float(box.get("ymin", 0.0)),
                "xmax": float(box.get("xmax", 0.0)),
                "ymax": float(box.get("ymax", 0.0)),
            }
        )
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open-vocabulary bootstrap labeling: detect with prompts, export YOLO labels, and write confidence CSV."
    )
    parser.add_argument("--images-dir", required=True, help="Folder of images to pseudo-label")
    parser.add_argument("--output-dir", required=True, help="Output folder for labels and CSV report")
    parser.add_argument("--classes", required=True, help="Comma-separated class names in desired YOLO class-id order")
    parser.add_argument(
        "--prompts-json",
        default=None,
        help="Optional JSON mapping class -> list of prompt strings (synonyms), e.g. {'lipstick':['lipstick','lip stick tube']}",
    )
    parser.add_argument("--model-id", default="google/owlv2-base-patch16-ensemble", help="Hugging Face model id")
    parser.add_argument("--min-conf", type=float, default=0.08, help="Minimum score kept from detector")
    parser.add_argument("--high-conf", type=float, default=0.40, help="Score threshold to auto-accept")
    parser.add_argument("--low-conf", type=float, default=0.20, help="Score threshold for review queue")
    parser.add_argument("--nms-iou", type=float, default=0.50, help="Class-wise NMS IoU threshold")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of processed images (0 = all)")
    args = parser.parse_args()

    if args.low_conf > args.high_conf:
        raise ValueError("--low-conf must be <= --high-conf")

    images_dir = Path(args.images_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = parse_classes(args.classes)
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    class_to_prompts, prompt_to_class = load_prompt_mapping(classes, args.prompts_json)

    prompt_list: List[str] = []
    for class_name in classes:
        prompt_list.extend(class_to_prompts[class_name])

    print("Loading model...")
    detector = pipeline(task="zero-shot-object-detection", model=args.model_id)

    image_paths = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    csv_path = output_dir / "confidence_report.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image_name",
                "image_path",
                "class_name",
                "class_id",
                "prompt",
                "score",
                "queue",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "x_center",
                "y_center",
                "width",
                "height",
            ],
        )
        writer.writeheader()

        accepted_count = 0
        review_count = 0
        discarded_count = 0

        for index, image_path in enumerate(image_paths, start=1):
            try:
                pil_image = PilImage.open(image_path).convert("RGB")
                width, height = pil_image.size

                predictions = detect_image(detector, pil_image, prompt_list, args.min_conf)
                mapped_detections: List[dict] = []

                for pred in predictions:
                    prompt = pred["prompt"]
                    class_name = prompt_to_class.get(prompt)
                    if class_name is None:
                        continue

                    class_id = class_to_id[class_name]
                    mapped = {
                        **pred,
                        "class_name": class_name,
                        "class_id": class_id,
                    }
                    mapped_detections.append(mapped)

                mapped_detections = classwise_nms(mapped_detections, args.nms_iou)

                yolo_lines: List[str] = []

                for det in mapped_detections:
                    x_center, y_center, box_w, box_h = xyxy_to_yolo(
                        det["xmin"], det["ymin"], det["xmax"], det["ymax"], width, height
                    )
                    if box_w <= 0 or box_h <= 0:
                        continue

                    queue = queue_for_score(det["score"], args.high_conf, args.low_conf)
                    if queue == "accept":
                        accepted_count += 1
                    elif queue == "review":
                        review_count += 1
                    else:
                        discarded_count += 1
                        continue

                    yolo_lines.append(
                        f"{det['class_id']} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
                    )

                    writer.writerow(
                        {
                            "image_name": image_path.name,
                            "image_path": str(image_path),
                            "class_name": det["class_name"],
                            "class_id": det["class_id"],
                            "prompt": det["prompt"],
                            "score": f"{det['score']:.6f}",
                            "queue": queue,
                            "xmin": f"{det['xmin']:.2f}",
                            "ymin": f"{det['ymin']:.2f}",
                            "xmax": f"{det['xmax']:.2f}",
                            "ymax": f"{det['ymax']:.2f}",
                            "x_center": f"{x_center:.6f}",
                            "y_center": f"{y_center:.6f}",
                            "width": f"{box_w:.6f}",
                            "height": f"{box_h:.6f}",
                        }
                    )

                label_path = labels_dir / f"{image_path.stem}.txt"
                with open(label_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(yolo_lines))

                if index % 25 == 0:
                    print(f"Processed {index}/{len(image_paths)} images")

            except Exception as exc:
                print(f"Skipping {image_path.name}: {exc}")

    classes_path = output_dir / "classes.txt"
    with open(classes_path, "w", encoding="utf-8") as f:
        for class_name in classes:
            f.write(class_name + "\n")

    print("\nDone")
    print(f"Images processed: {len(image_paths)}")
    print(f"Accepted detections: {accepted_count}")
    print(f"Review detections: {review_count}")
    print(f"Discarded detections: {discarded_count}")
    print(f"Labels: {labels_dir}")
    print(f"Confidence CSV: {csv_path}")
    print(f"Classes file: {classes_path}")


if __name__ == "__main__":
    main()
