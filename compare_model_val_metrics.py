from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

CONFIDENCE_THRESH = 0.3
LINE_THICKNESS = 2
FONT_SCALE = 0.65

'''
python compare_model_val_metrics.py \
  --v8-weights /Users/michael.mandiberg/Documents/GitHub/takingstock/models/takingstock_head_heart_v4_yolov8x/weights/best.pt \
  --y26-weights /Users/michael.mandiberg/Documents/GitHub/takingstock/models/takingstock_head_heart_v4_yolo26x/weights/best.pt \
  --y26obj-weights /Users/michael.mandiberg/Documents/GitHub/takingstock/models/takingstock_head_heart_v4_yolo26xObj/weights/best.pt \
  --data /Users/michael.mandiberg/Documents/GitHub/taking-stock-yolo/yolo_dataset/data.yaml \
  --device mps \
  --imgsz 640 \
  --batch 16 \
  --split val \
  --output-json /Users/michael.mandiberg/Documents/GitHub/taking-stock-yolo/runs/compare_v8_y26_y26obj.json
  
  '''

def evaluate_model(
    weights: str,
    data_yaml: str,
    device: str,
    imgsz: int,
    batch: int,
    split: str,
    iou: float,
    conf: float,
) -> dict:
    def to_float(value):
        try:
            return float(value)
        except Exception:
            pass
        try:
            if hasattr(value, "item"):
                return float(value.item())
        except Exception:
            pass
        return None

    model = YOLO(weights)
    metrics = model.val(
        data=data_yaml,
        device=device,
        imgsz=imgsz,
        batch=batch,
        split=split,
        iou=iou,
        conf=conf,
        verbose=False,
    )

    cls_loss = None
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        for key, value in results_dict.items():
            key_l = str(key).lower()
            if "cls_loss" in key_l or ("loss" in key_l and "cls" in key_l):
                cls_loss = to_float(value)
                break

    if cls_loss is None:
        for attr_name in dir(metrics):
            attr_l = attr_name.lower()
            if "cls" in attr_l and "loss" in attr_l:
                cls_loss = to_float(getattr(metrics, attr_name, None))
                if cls_loss is not None:
                    break

    if cls_loss is None and hasattr(metrics, "loss"):
        loss_value = getattr(metrics, "loss")
        try:
            if hasattr(loss_value, "tolist"):
                loss_list = loss_value.tolist()
            else:
                loss_list = list(loss_value)
            if len(loss_list) >= 2:
                cls_loss = to_float(loss_list[1])
        except Exception:
            cls_loss = None

    names = {}
    if hasattr(metrics, "names") and isinstance(metrics.names, dict):
        names = {int(k): v for k, v in metrics.names.items()}

    box_maps = []
    if hasattr(metrics, "box") and hasattr(metrics.box, "maps"):
        box_maps = list(metrics.box.maps)

    per_class = []
    for class_index, class_map in enumerate(box_maps):
        per_class.append(
            {
                "class_id": class_index,
                "class_name": names.get(class_index, str(class_index)),
                "map50_95": float(class_map),
            }
        )

    summary = {
        "weights": weights,
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "cls_loss": cls_loss,
        "per_class": per_class,
    }
    return summary


def print_summary(label: str, result: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"weights:   {result['weights']}")
    print(f"mAP50-95: {result['map50_95']:.6f}")
    print(f"mAP50:    {result['map50']:.6f}")
    print(f"mAP75:    {result['map75']:.6f}")
    if result.get("cls_loss") is None:
        print("cls_loss: n/a (not reported by this validation run)")
    else:
        print(f"cls_loss: {result['cls_loss']:.6f}")
    if result.get("class_loss_proxy") is not None:
        print(f"class_loss_proxy: {result['class_loss_proxy']:.6f}")
    if result.get("class_accuracy_proxy") is not None:
        print(f"class_accuracy_proxy: {result['class_accuracy_proxy']:.6f}")


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_split_images(data_yaml: str, split: str) -> list[Path]:
    data_yaml_path = Path(data_yaml).resolve()
    data_cfg = yaml.safe_load(data_yaml_path.read_text())

    cfg_root = data_yaml_path.parent
    data_root = Path(data_cfg.get("path", cfg_root))
    if not data_root.is_absolute():
        data_root = (cfg_root / data_root).resolve()

    split_value = data_cfg.get(split)
    if split_value is None:
        raise ValueError(f"Split '{split}' was not found in {data_yaml}")

    split_entries: Iterable[str]
    if isinstance(split_value, list):
        split_entries = split_value
    else:
        split_entries = [split_value]

    images: list[Path] = []
    for entry in split_entries:
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = (data_root / candidate).resolve()

        if candidate.is_dir():
            images.extend(sorted(p for p in candidate.rglob("*") if p.is_file() and is_image_file(p)))
        elif candidate.is_file() and candidate.suffix.lower() == ".txt":
            for line in candidate.read_text().splitlines():
                value = line.strip()
                if not value:
                    continue
                img_path = Path(value)
                if not img_path.is_absolute():
                    img_path = (data_root / img_path).resolve()
                if img_path.is_file() and is_image_file(img_path):
                    images.append(img_path)
        elif candidate.is_file() and is_image_file(candidate):
            images.append(candidate)

    unique_sorted = sorted({p.resolve() for p in images})
    return unique_sorted


def image_to_label_path(image_path: Path) -> Path:
    image_str = str(image_path)
    if f"{Path('/').as_posix()}images{Path('/').as_posix()}" in image_str:
        label_str = image_str.replace(f"{Path('/').as_posix()}images{Path('/').as_posix()}", f"{Path('/').as_posix()}labels{Path('/').as_posix()}")
        return Path(label_str).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def load_gt_labels(image_path: Path, image_w: int, image_h: int) -> list[dict]:
    label_path = image_to_label_path(image_path)
    if not label_path.exists():
        return []

    gt = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:])
        except ValueError:
            continue
        x1 = (xc - bw / 2.0) * image_w
        y1 = (yc - bh / 2.0) * image_h
        x2 = (xc + bw / 2.0) * image_w
        y2 = (yc + bh / 2.0) * image_h
        gt.append({"class_id": class_id, "xyxy": [x1, y1, x2, y2]})
    return gt


def iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def evaluate_image_matches(gt: list[dict], preds: list[dict], match_iou: float) -> dict:
    used_pred = set()
    correct = 0
    misclassified = 0
    missed = 0

    for gt_item in gt:
        best_idx = -1
        best_iou = -1.0
        for pred_idx, pred_item in enumerate(preds):
            if pred_idx in used_pred:
                continue
            current_iou = iou_xyxy(gt_item["xyxy"], pred_item["xyxy"])
            if current_iou > best_iou:
                best_iou = current_iou
                best_idx = pred_idx

        if best_idx >= 0 and best_iou >= match_iou:
            used_pred.add(best_idx)
            if preds[best_idx]["class_id"] == gt_item["class_id"]:
                correct += 1
            else:
                misclassified += 1
        else:
            missed += 1

    is_correct = (misclassified == 0 and missed == 0)
    return {
        "is_correct": is_correct,
        "correct": correct,
        "misclassified": misclassified,
        "missed": missed,
        "total_gt": len(gt),
    }


def evaluate_class_focus(gt: list[dict], preds: list[dict]) -> dict:
    pred_class_counts = {}
    for pred in preds:
        class_id = pred["class_id"]
        pred_class_counts[class_id] = pred_class_counts.get(class_id, 0) + 1

    class_hits = 0
    class_missed = 0
    for gt_item in gt:
        class_id = gt_item["class_id"]
        if pred_class_counts.get(class_id, 0) > 0:
            class_hits += 1
            pred_class_counts[class_id] -= 1
        else:
            class_missed += 1

    is_class_correct = class_missed == 0
    return {
        "is_class_correct": is_class_correct,
        "class_hits": class_hits,
        "class_missed": class_missed,
        "total_gt": len(gt),
    }


def draw_panel(image: np.ndarray, gt: list[dict], preds: list[dict], names: dict[int, str], title: str, is_class_correct: bool) -> np.ndarray:
    canvas = image.copy()
    h, w = canvas.shape[:2]

    for gt_item in gt:
        x1, y1, x2, y2 = map(int, gt_item["xyxy"])
        cls_name = names.get(gt_item["class_id"], str(gt_item["class_id"]))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 200, 0), LINE_THICKNESS)
        cv2.putText(canvas, f"GT:{cls_name}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 200, 0), LINE_THICKNESS, cv2.LINE_AA)

    for pred in preds:
        x1, y1, x2, y2 = map(int, pred["xyxy"])
        cls_name = names.get(pred["class_id"], str(pred["class_id"]))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (120, 255, 120), LINE_THICKNESS)
        cv2.putText(
            canvas,
            f"P:{cls_name} {pred['conf']:.2f}",
            (x1, min(h - 10, max(22, y2 + 20))),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (120, 255, 120),
            LINE_THICKNESS,
            cv2.LINE_AA,
        )

    cv2.putText(canvas, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), LINE_THICKNESS, cv2.LINE_AA)
    if is_class_correct:
        cv2.circle(canvas, (22, h - 22), 10, (0, 200, 0), -1)
        cv2.circle(canvas, (22, h - 22), 10, (255, 255, 255), LINE_THICKNESS)

    return canvas


def save_incorrect_comparisons(
    args,
    model_v8: YOLO,
    model_y26: YOLO,
    model_y26obj: YOLO,
    names: dict[int, str],
) -> tuple[int, Path, list[dict], dict]:
    image_paths = resolve_split_images(args.data, args.split)
    out_dir = Path(args.incorrect_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_specs = [
        ("v8", model_v8),
        ("y26", model_y26),
        ("y26obj", model_y26obj),
    ]

    mistakes = []
    saved_count = 0
    class_focus_stats = {
        "v8": {"total_gt": 0, "class_hits": 0, "class_missed": 0, "images_total": 0, "images_class_correct": 0},
        "y26": {"total_gt": 0, "class_hits": 0, "class_missed": 0, "images_total": 0, "images_class_correct": 0},
        "y26obj": {"total_gt": 0, "class_hits": 0, "class_missed": 0, "images_total": 0, "images_class_correct": 0},
    }

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        gt = load_gt_labels(image_path, w, h)
        if not gt:
            continue

        per_model_preds = {}
        per_model_eval = {}
        per_model_class_eval = {}
        any_incorrect = False

        for model_name, model in model_specs:
            results = model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                device=args.device,
                conf=args.conf,
                iou=args.iou,
                verbose=False,
            )
            result = results[0]
            boxes = result.boxes
            pred_items = []
            if boxes is not None and boxes.xyxy is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                cls_arr = boxes.cls.cpu().numpy().astype(int)
                conf_arr = boxes.conf.cpu().numpy()
                for idx in range(len(xyxy)):
                    conf_value = float(conf_arr[idx])
                    if conf_value < CONFIDENCE_THRESH:
                        continue
                    pred_items.append(
                        {
                            "class_id": int(cls_arr[idx]),
                            "conf": conf_value,
                            "xyxy": xyxy[idx].tolist(),
                        }
                    )

            eval_result = evaluate_image_matches(gt, pred_items, args.match_iou)
            class_eval_result = evaluate_class_focus(gt, pred_items)
            per_model_preds[model_name] = pred_items
            per_model_eval[model_name] = eval_result
            per_model_class_eval[model_name] = class_eval_result
            class_focus_stats[model_name]["total_gt"] += class_eval_result["total_gt"]
            class_focus_stats[model_name]["class_hits"] += class_eval_result["class_hits"]
            class_focus_stats[model_name]["class_missed"] += class_eval_result["class_missed"]
            class_focus_stats[model_name]["images_total"] += 1
            if class_eval_result["is_class_correct"]:
                class_focus_stats[model_name]["images_class_correct"] += 1
            if not eval_result["is_correct"]:
                any_incorrect = True

        if not any_incorrect:
            continue

        panels = []
        panels.append(draw_panel(image, gt, per_model_preds["v8"], names, "YOLOv8", per_model_class_eval["v8"]["is_class_correct"]))
        panels.append(draw_panel(image, gt, per_model_preds["y26"], names, "YOLO26", per_model_class_eval["y26"]["is_class_correct"]))
        panels.append(draw_panel(image, gt, per_model_preds["y26obj"], names, "YOLO26-Obj365", per_model_class_eval["y26obj"]["is_class_correct"]))

        combined = cv2.hconcat(panels)
        out_path = out_dir / f"{image_path.stem}_compare.jpg"
        cv2.imwrite(str(out_path), combined)
        saved_count += 1

        mistakes.append(
            {
                "image": str(image_path),
                "output": str(out_path),
                "v8": per_model_eval["v8"],
                "y26": per_model_eval["y26"],
                "y26obj": per_model_eval["y26obj"],
                "v8_class_focus": per_model_class_eval["v8"],
                "y26_class_focus": per_model_class_eval["y26"],
                "y26obj_class_focus": per_model_class_eval["y26obj"],
            }
        )

    return saved_count, out_dir, mistakes, class_focus_stats


def per_class_to_dict(result: dict) -> dict[int, dict]:
    return {entry["class_id"]: entry for entry in result.get("per_class", [])}


def build_per_class_comparison(v8: dict, y26: dict, y26obj: dict) -> list[dict]:
    v8_by_id = per_class_to_dict(v8)
    y26_by_id = per_class_to_dict(y26)
    y26obj_by_id = per_class_to_dict(y26obj)
    all_class_ids = sorted(set(v8_by_id) | set(y26_by_id) | set(y26obj_by_id))

    rows = []
    for class_id in all_class_ids:
        v8_entry = v8_by_id.get(class_id)
        y26_entry = y26_by_id.get(class_id)
        y26obj_entry = y26obj_by_id.get(class_id)

        class_name = str(class_id)
        for entry in (v8_entry, y26_entry, y26obj_entry):
            if entry is not None and entry.get("class_name"):
                class_name = entry["class_name"]
                break

        v8_map = float(v8_entry["map50_95"]) if v8_entry is not None else 0.0
        y26_map = float(y26_entry["map50_95"]) if y26_entry is not None else 0.0
        y26obj_map = float(y26obj_entry["map50_95"]) if y26obj_entry is not None else 0.0

        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "v8_map50_95": v8_map,
                "y26_map50_95": y26_map,
                "y26obj_map50_95": y26obj_map,
                "delta_y26_vs_v8": y26_map - v8_map,
                "delta_y26obj_vs_v8": y26obj_map - v8_map,
                "delta_y26obj_vs_y26": y26obj_map - y26_map,
            }
        )
    return rows


def print_per_class_comparison(rows: list[dict]) -> None:
    print("\n=== Per-class mAP50-95 Comparison ===")
    print("class_id class_name                 v8       y26      y26obj   d(y26-v8) d(y26obj-v8) d(y26obj-y26)")
    for row in rows:
        print(
            f"{row['class_id']:>7} "
            f"{row['class_name'][:24]:<24} "
            f"{row['v8_map50_95']:>8.4f} "
            f"{row['y26_map50_95']:>8.4f} "
            f"{row['y26obj_map50_95']:>8.4f} "
            f"{row['delta_y26_vs_v8']:>+10.4f} "
            f"{row['delta_y26obj_vs_v8']:>+13.4f} "
            f"{row['delta_y26obj_vs_y26']:>+14.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare three YOLO models on the same validation split."
    )
    parser.add_argument("--v8-weights", required=True, help="Path to YOLOv8 best.pt")
    parser.add_argument("--y26-weights", required=True, help="Path to YOLO26 best.pt")
    parser.add_argument("--y26obj-weights", required=True, help="Path to YOLO26 Objects365-initialized best.pt")
    parser.add_argument("--data", required=True, help="Path to dataset YAML")
    parser.add_argument("--device", default="mps", help="Device: mps, cuda, cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--split", default="val", help="Dataset split: train/val/test")
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--match-iou", type=float, default=0.5, help="IoU threshold for GT-to-pred match when flagging misclassified/missed objects")
    parser.add_argument(
        "--incorrect-out-dir",
        default="/Users/michael.mandiberg/Documents/GitHub/taking-stock-yolo/runs/incorrect_images_compare",
        help="Directory to save side-by-side comparison images for mistakes",
    )
    parser.add_argument(
        "--skip-incorrect-images",
        action="store_true",
        help="Skip saving per-image mistake comparison grids",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save full comparison JSON",
    )

    args = parser.parse_args()

    v8 = evaluate_model(
        weights=args.v8_weights,
        data_yaml=args.data,
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
        split=args.split,
        iou=args.iou,
        conf=args.conf,
    )
    y26 = evaluate_model(
        weights=args.y26_weights,
        data_yaml=args.data,
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
        split=args.split,
        iou=args.iou,
        conf=args.conf,
    )
    y26obj = evaluate_model(
        weights=args.y26obj_weights,
        data_yaml=args.data,
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
        split=args.split,
        iou=args.iou,
        conf=args.conf,
    )

    print_summary("YOLOv8", v8)
    print_summary("YOLO26", y26)
    print_summary("YOLO26-Obj365", y26obj)
    print(f"\nUsing CONFIDENCE_THRESH={CONFIDENCE_THRESH:.2f} for class-focused comparison overlays/evaluation")

    model_v8 = YOLO(args.v8_weights)
    model_y26 = YOLO(args.y26_weights)
    model_y26obj = YOLO(args.y26obj_weights)

    names = {entry["class_id"]: entry["class_name"] for entry in v8.get("per_class", [])}
    for model_result in (y26, y26obj):
        for entry in model_result.get("per_class", []):
            names.setdefault(entry["class_id"], entry["class_name"])

    delta_y26_vs_v8 = {
        "map50_95": y26["map50_95"] - v8["map50_95"],
        "map50": y26["map50"] - v8["map50"],
        "map75": y26["map75"] - v8["map75"],
    }
    delta_y26obj_vs_v8 = {
        "map50_95": y26obj["map50_95"] - v8["map50_95"],
        "map50": y26obj["map50"] - v8["map50"],
        "map75": y26obj["map75"] - v8["map75"],
    }
    delta_y26obj_vs_y26 = {
        "map50_95": y26obj["map50_95"] - y26["map50_95"],
        "map50": y26obj["map50"] - y26["map50"],
        "map75": y26obj["map75"] - y26["map75"],
    }

    print("\n=== Delta (YOLO26 - YOLOv8) ===")
    print(f"mAP50-95: {delta_y26_vs_v8['map50_95']:+.6f}")
    print(f"mAP50:    {delta_y26_vs_v8['map50']:+.6f}")
    print(f"mAP75:    {delta_y26_vs_v8['map75']:+.6f}")

    print("\n=== Delta (YOLO26-Obj365 - YOLOv8) ===")
    print(f"mAP50-95: {delta_y26obj_vs_v8['map50_95']:+.6f}")
    print(f"mAP50:    {delta_y26obj_vs_v8['map50']:+.6f}")
    print(f"mAP75:    {delta_y26obj_vs_v8['map75']:+.6f}")

    print("\n=== Delta (YOLO26-Obj365 - YOLO26) ===")
    print(f"mAP50-95: {delta_y26obj_vs_y26['map50_95']:+.6f}")
    print(f"mAP50:    {delta_y26obj_vs_y26['map50']:+.6f}")
    print(f"mAP75:    {delta_y26obj_vs_y26['map75']:+.6f}")

    per_class_comparison = build_per_class_comparison(v8, y26, y26obj)
    print_per_class_comparison(per_class_comparison)

    incorrect_image_summary = {
        "saved_count": 0,
        "output_dir": args.incorrect_out_dir,
        "images": [],
    }
    class_focus_stats = {
        "v8": {"total_gt": 0, "class_hits": 0, "class_missed": 0, "images_total": 0, "images_class_correct": 0},
        "y26": {"total_gt": 0, "class_hits": 0, "class_missed": 0, "images_total": 0, "images_class_correct": 0},
        "y26obj": {"total_gt": 0, "class_hits": 0, "class_missed": 0, "images_total": 0, "images_class_correct": 0},
    }
    if not args.skip_incorrect_images:
        saved_count, out_dir, mistakes, class_focus_stats = save_incorrect_comparisons(
            args=args,
            model_v8=model_v8,
            model_y26=model_y26,
            model_y26obj=model_y26obj,
            names=names,
        )
        incorrect_image_summary = {
            "saved_count": saved_count,
            "output_dir": str(out_dir),
            "images": mistakes,
        }
        print(f"\nSaved {saved_count} incorrect-image comparisons to: {out_dir}")

    def apply_class_focus_proxy(result: dict, stats: dict) -> None:
        total_gt = stats.get("total_gt", 0)
        class_missed = stats.get("class_missed", 0)
        class_hits = stats.get("class_hits", 0)
        if total_gt > 0:
            result["class_loss_proxy"] = class_missed / total_gt
            result["class_accuracy_proxy"] = class_hits / total_gt
        else:
            result["class_loss_proxy"] = None
            result["class_accuracy_proxy"] = None

    apply_class_focus_proxy(v8, class_focus_stats["v8"])
    apply_class_focus_proxy(y26, class_focus_stats["y26"])
    apply_class_focus_proxy(y26obj, class_focus_stats["y26obj"])

    print("\n=== Class-Focused Proxy (all GT objects) ===")
    for label, stats, result in (
        ("YOLOv8", class_focus_stats["v8"], v8),
        ("YOLO26", class_focus_stats["y26"], y26),
        ("YOLO26-Obj365", class_focus_stats["y26obj"], y26obj),
    ):
        loss_proxy = result.get("class_loss_proxy")
        acc_proxy = result.get("class_accuracy_proxy")
        if loss_proxy is None or acc_proxy is None:
            print(f"{label}: class_loss_proxy=n/a class_accuracy_proxy=n/a")
            continue
        print(
            f"{label}: class_loss_proxy={loss_proxy:.6f} class_accuracy_proxy={acc_proxy:.6f} "
            f"(hits={stats['class_hits']} missed={stats['class_missed']} total_gt={stats['total_gt']})"
        )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "data": args.data,
            "device": args.device,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "split": args.split,
            "iou": args.iou,
            "conf": args.conf,
            "v8": v8,
            "y26": y26,
            "y26obj": y26obj,
            "delta_y26_vs_v8": delta_y26_vs_v8,
            "delta_y26obj_vs_v8": delta_y26obj_vs_v8,
            "delta_y26obj_vs_y26": delta_y26obj_vs_y26,
            "per_class_comparison": per_class_comparison,
            "incorrect_image_summary": incorrect_image_summary,
            "class_focus_stats": class_focus_stats,
            "confidence_thresh": CONFIDENCE_THRESH,
        }
        output_path.write_text(json.dumps(output, indent=2))
        print(f"\nSaved comparison JSON to: {output_path}")


if __name__ == "__main__":
    main()
