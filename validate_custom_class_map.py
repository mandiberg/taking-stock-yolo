#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter

from class_map_utils import load_class_records


def main() -> None:
    records = load_class_records()

    ids = [int(row["id"]) for row in records]
    names = [str(row["name"]) for row in records]

    id_counts = Counter(ids)
    name_counts = Counter(name.lower() for name in names)

    duplicate_ids = sorted([cid for cid, count in id_counts.items() if count > 1])
    duplicate_names = sorted([name for name, count in name_counts.items() if count > 1])

    print("=== Custom Class Map Validation ===")
    print(f"total classes: {len(records)}")
    print(f"min id: {min(ids) if ids else 'n/a'}")
    print(f"max id: {max(ids) if ids else 'n/a'}")
    print(f"duplicate ids: {duplicate_ids}")
    print(f"duplicate names: {duplicate_names}")

    if duplicate_ids or duplicate_names:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
