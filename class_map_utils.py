from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_CLASS_MAP_PATH = Path(__file__).resolve().parent / "config" / "custom_class_map.json"


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def load_class_records(class_map_path: Path | None = None) -> list[dict]:
    # 1. Fall back to default path if None
    raw_path = class_map_path or DEFAULT_CLASS_MAP_PATH
    
    # 2. Ensure it is a Path object, even if a string was passed
    path = Path(raw_path) 
    
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    classes = payload.get("classes", [])
    if not isinstance(classes, list):
        raise ValueError(f"Invalid classes payload in {path}")

    return classes


def get_id_to_name(
    class_map_path: Path | None = None,
    min_id: int | None = None,
    max_id: int | None = None,
) -> dict[int, str]:
    id_to_name: dict[int, str] = {}
    for row in load_class_records(class_map_path):
        class_id = int(row["id"])
        if min_id is not None and class_id < min_id:
            continue
        if max_id is not None and class_id > max_id:
            continue
        id_to_name[class_id] = str(row["name"])
    return id_to_name


def get_name_to_id(class_map_path: Path | None = None) -> dict[str, int]:
    name_to_id: dict[str, int] = {}
    for row in load_class_records(class_map_path):
        class_id = int(row["id"])
        canonical = str(row["name"])

        candidate_names = [canonical]
        aliases = row.get("aliases", [])
        if isinstance(aliases, list):
            candidate_names.extend(str(alias) for alias in aliases)

        for candidate in candidate_names:
            name_to_id[_normalize_token(candidate)] = class_id

    return name_to_id


def resolve_class_name_to_id(name: str, class_map_path: Path | None = None) -> int | None:
    lookup = get_name_to_id(class_map_path)
    return lookup.get(_normalize_token(name))
