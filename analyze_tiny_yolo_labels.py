#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import median

IGNORED_CLASS_PAIR = {89, 90}
IGNORE_FOLDERS = {"reprocess", "depricated"}


@dataclass
class FileStats:
    file_path: str
    annotation_count: int
    smallest_area: float
    next_smallest_area: float
    delta: float
    ratio_small_to_next: float
    smallest_class_id: int
    next_smallest_class_id: int
    smallest_xc: float
    smallest_yc: float
    smallest_w: float
    smallest_h: float
    next_xc: float
    next_yc: float
    next_w: float
    next_h: float
    smallest_aspect_ratio: float
    min_corner_dist: float
    shared_corner_like: bool
    bottom_left_id_like: bool
    low_conf: bool = False
    high_conf: bool = False


def parse_label_line(line: str):
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:])
        area = max(0.0, w) * max(0.0, h)
        return class_id, xc, yc, w, h, area
    except ValueError:
        return None


def corners_xywh(xc: float, yc: float, w: float, h: float) -> list[tuple[float, float]]:
    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0
    return [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]


def min_corner_distance(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    a_corners = corners_xywh(*box_a)
    b_corners = corners_xywh(*box_b)
    best = float("inf")
    for ax, ay in a_corners:
        for bx, by in b_corners:
            distance = math.hypot(ax - bx, ay - by)
            if distance < best:
                best = distance
    return best


def analyze_label_file(path: Path) -> FileStats | None:
    entries = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parsed = parse_label_line(line)
        if parsed is not None:
            entries.append(parsed)

    if len(entries) < 2:
        return None

    entries_sorted = sorted(entries, key=lambda x: x[5])
    smallest_class_id, smallest_xc, smallest_yc, smallest_w, smallest_h, smallest_area = entries_sorted[0]
    next_class_id, next_xc, next_yc, next_w, next_h, next_smallest_area = entries_sorted[1]

    delta = next_smallest_area - smallest_area
    if next_smallest_area > 0:
        ratio = smallest_area / next_smallest_area
    else:
        ratio = 0.0

    smallest_aspect_ratio = (smallest_w / smallest_h) if smallest_h > 0 else float("inf")
    corner_dist = min_corner_distance(
        (smallest_xc, smallest_yc, smallest_w, smallest_h),
        (next_xc, next_yc, next_w, next_h),
    )
    shared_corner_like = corner_dist <= 0.01
    bottom_left_id_like = (
        smallest_xc <= 0.30
        and smallest_yc >= 0.75
        and smallest_aspect_ratio >= 2.5
        and smallest_area <= 0.02
    )

    return FileStats(
        file_path=str(path),
        annotation_count=len(entries),
        smallest_area=smallest_area,
        next_smallest_area=next_smallest_area,
        delta=delta,
        ratio_small_to_next=ratio,
        smallest_class_id=smallest_class_id,
        next_smallest_class_id=next_class_id,
        smallest_xc=smallest_xc,
        smallest_yc=smallest_yc,
        smallest_w=smallest_w,
        smallest_h=smallest_h,
        next_xc=next_xc,
        next_yc=next_yc,
        next_w=next_w,
        next_h=next_h,
        smallest_aspect_ratio=smallest_aspect_ratio,
        min_corner_dist=corner_dist,
        shared_corner_like=shared_corner_like,
        bottom_left_id_like=bottom_left_id_like,
    )


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values_sorted = sorted(values)
    idx = (len(values_sorted) - 1) * q
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return values_sorted[lower]
    frac = idx - lower
    return values_sorted[lower] * (1 - frac) + values_sorted[upper] * frac


def bucket_name(count: int) -> str:
    if count <= 3:
        return str(count)
    if count <= 5:
        return "4-5"
    return ">5"


def is_ignored_pair(s: FileStats) -> bool:
    return {s.smallest_class_id, s.next_smallest_class_id} == IGNORED_CLASS_PAIR


def is_in_ignored_folder(path: Path) -> bool:
    return any(part in IGNORE_FOLDERS for part in path.parts)


def is_low_conf(stats: FileStats, max_ratio: float, max_smallest_area: float) -> bool:
    return (
        stats.ratio_small_to_next <= max_ratio
        and stats.smallest_area <= max_smallest_area
        and not is_ignored_pair(stats)
    )


def is_high_conf(
    stats: FileStats,
    high_conf_ratio: float,
    high_conf_smallest_area: float,
) -> bool:
    if not stats.low_conf:
        return False
    return (
        stats.ratio_small_to_next <= high_conf_ratio
        or stats.smallest_area <= high_conf_smallest_area
        or stats.shared_corner_like
        or stats.bottom_left_id_like
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze YOLO label files for suspicious tiny accidental annotations."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/Users/michael.mandiberg/Documents/YOLO_Training_Data"),
        help="Root directory to scan recursively for *.txt YOLO label files.",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=0.2,
        help="Flag as suspicious when smallest_area / next_smallest_area <= this value.",
    )
    parser.add_argument(
        "--max-smallest-area",
        type=float,
        default=0.008,
        help="Flag as suspicious when smallest area <= this value (normalized area).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of most suspicious files to print.",
    )
    parser.add_argument(
        "--high-conf-ratio",
        type=float,
        default=0.08,
        help="High confidence trigger: ratio_small_to_next <= this value.",
    )
    parser.add_argument(
        "--high-conf-smallest-area",
        type=float,
        default=0.003,
        help="High confidence trigger: smallest area <= this value.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save full analysis JSON.",
    )

    args = parser.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(f"Root path not found: {args.root}")

    all_txt = [p for p in args.root.rglob("*.txt") if not is_in_ignored_folder(p)]
    analyzed: list[FileStats] = []

    for txt in all_txt:
        stats = analyze_label_file(txt)
        if stats is not None:
            analyzed.append(stats)

    analyzed.sort(key=lambda s: (s.ratio_small_to_next, s.smallest_area, -s.annotation_count))

    for s in analyzed:
        s.low_conf = is_low_conf(s, args.max_ratio, args.max_smallest_area)
        s.high_conf = is_high_conf(s, args.high_conf_ratio, args.high_conf_smallest_area)

    low_conf = [s for s in analyzed if s.low_conf]
    high_conf = [s for s in analyzed if s.high_conf]

    by_bucket = defaultdict(list)
    for s in analyzed:
        by_bucket[bucket_name(s.annotation_count)].append(s)

    print("=== Tiny Label Analysis ===")
    print(f"Root: {args.root}")
    print(f"Total .txt files scanned: {len(all_txt)}")
    print(f"Files with >=2 valid annotations: {len(analyzed)}")
    print(f"low_conf files (ratio <= {args.max_ratio} and smallest_area <= {args.max_smallest_area}): {len(low_conf)}")
    print(
        "high_conf files (low_conf plus [ratio<="
        f"{args.high_conf_ratio} or smallest_area<={args.high_conf_smallest_area} "
        "or shared_corner_like or bottom_left_id_like]): "
        f"{len(high_conf)}"
    )
    print(
        f"high_conf feature hits: shared_corner_like={sum(1 for s in high_conf if s.shared_corner_like)} "
        f"bottom_left_id_like={sum(1 for s in high_conf if s.bottom_left_id_like)}"
    )

    if analyzed:
        ratios = [s.ratio_small_to_next for s in analyzed]
        smallest_areas = [s.smallest_area for s in analyzed]
        print("\n--- Global distribution ---")
        print(f"ratio smallest/next : median={median(ratios):.4f} p10={quantile(ratios, 0.10):.4f} p05={quantile(ratios, 0.05):.4f} p01={quantile(ratios, 0.01):.4f}")
        print(f"smallest area       : median={median(smallest_areas):.6f} p10={quantile(smallest_areas, 0.10):.6f} p05={quantile(smallest_areas, 0.05):.6f} p01={quantile(smallest_areas, 0.01):.6f}")

    print("\n--- By annotation count bucket ---")
    for bucket in ["2", "3", "4-5", ">5"]:
        rows = by_bucket.get(bucket, [])
        if not rows:
            print(f"{bucket:>3}: count=0")
            continue
        ratios = [r.ratio_small_to_next for r in rows]
        small = [r.smallest_area for r in rows]
        low_count = sum(1 for r in rows if r.low_conf)
        high_count = sum(1 for r in rows if r.high_conf)
        print(
            f"{bucket:>3}: count={len(rows):5d} low_conf={low_count:4d} high_conf={high_count:4d} "
            f"median_ratio={median(ratios):.4f} p05_ratio={quantile(ratios, 0.05):.4f} "
            f"median_smallest_area={median(small):.6f}"
        )

    ranked_low = sorted([s for s in low_conf], key=lambda s: (s.ratio_small_to_next, s.smallest_area, -s.annotation_count))
    ranked_high = sorted([s for s in high_conf], key=lambda s: (s.ratio_small_to_next, s.smallest_area, -s.annotation_count))

    print(f"\n--- Top {min(args.top_n, len(ranked_high))} high_conf files ---")
    for i, s in enumerate(ranked_high[: args.top_n], start=1):
        print(
            f"{i:3d}. n={s.annotation_count:2d} ratio={s.ratio_small_to_next:.4f} "
            f"small={s.smallest_area:.6f} next={s.next_smallest_area:.6f} delta={s.delta:.6f} "
            f"corner={s.min_corner_dist:.4f} shared={int(s.shared_corner_like)} bl={int(s.bottom_left_id_like)} "
            f"cls={s.smallest_class_id}->{s.next_smallest_class_id} {s.file_path}"
        )

    print(f"\n--- Top {min(args.top_n, len(ranked_low))} low_conf files ---")
    for i, s in enumerate(ranked_low[: args.top_n], start=1):
        print(
            f"{i:3d}. n={s.annotation_count:2d} ratio={s.ratio_small_to_next:.4f} "
            f"small={s.smallest_area:.6f} next={s.next_smallest_area:.6f} delta={s.delta:.6f} "
            f"corner={s.min_corner_dist:.4f} shared={int(s.shared_corner_like)} bl={int(s.bottom_left_id_like)} "
            f"cls={s.smallest_class_id}->{s.next_smallest_class_id} {s.file_path}"
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "root": str(args.root),
            "total_txt_files": len(all_txt),
            "files_with_2plus_annotations": len(analyzed),
            "low_conf_count": len(low_conf),
            "high_conf_count": len(high_conf),
            "max_ratio": args.max_ratio,
            "max_smallest_area": args.max_smallest_area,
            "high_conf_ratio": args.high_conf_ratio,
            "high_conf_smallest_area": args.high_conf_smallest_area,
            "ignored_class_pair": sorted(IGNORED_CLASS_PAIR),
            "rows": [asdict(row) for row in analyzed],
        }
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved JSON report: {args.output_json}")


if __name__ == "__main__":
    main()
