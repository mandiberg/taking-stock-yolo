import os
import json
from html import escape

# --- inputs ---
CLASS_MAP_PATH = "flags/class_map.json"
OUTPUT_XML_PATH = "flags/labelstudio_label_config.xml"

# Most-used / top priority (IDs from your notes + EU/UN)
HIGH_PRIORITY_IDS = [
    117, 226, 88, 211, 27, 228, 212, 35, 225, 227, 213, 72, 163
]

# Medium cohort by class IDs only
# peru=154, france=67, china=41, italy=94, jamaica=96,
# israel=93, slovenia=180, spain=187, north korea=145, greece=75, liberia=108
MEDIUM_PRIORITY_IDS = [
    154, 67, 41, 94, 96,
    93, 180, 187, 145, 75, 108,
    11, 205, 149, 193, 157,
    18, 124, 92, 8,
]

# Vivid, highly distinguishable
HIGH_COLORS = [
    "#E53935", "#8E24AA", "#1E88E5", "#00ACC1", "#43A047",
    "#FB8C00", "#3949AB", "#F4511E", "#00897B", "#6D4C41",
    "#D81B60", "#5E35B1", "#7CB342"
]

# Medium saturation
MED_COLORS = [
    "#4C78A8", "#59A14F", "#9C755F", "#76B7B2", "#EDC948",
    "#B07AA1", "#86BCB6", "#A0CBE8", "#8CD17D", "#F1CE63", "#D4A6C8"
]

# Light / desaturated low-priority palette
LOW_COLORS = [
    "#DCE6F2", "#E2EBD8", "#EAE2F1", "#F2E9DD", "#DCEFEA",
    "#E9EDF7", "#F0E8F3", "#EAF0E2", "#EDEDED", "#E3EEF6"
]

def build():
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # include-only classes with numeric ids
    classes = [
        c for c in data["classes"]
        if c.get("include") is True and isinstance(c.get("yolo_class_id"), int)
    ]

    by_id = {c["yolo_class_id"]: c for c in classes}
    used_ids = set()

    # 1) high priority by explicit IDs
    high = []
    for cid in HIGH_PRIORITY_IDS:
        if cid in by_id:
            high.append(by_id[cid])
            used_ids.add(cid)

    # 2) medium priority by explicit IDs
    medium = []
    for cid in MEDIUM_PRIORITY_IDS:
        if cid in by_id and cid not in used_ids:
            medium.append(by_id[cid])
            used_ids.add(cid)

    # 3) all remaining classes by ID
    low = [c for c in classes if c["yolo_class_id"] not in used_ids]
    low.sort(key=lambda x: x["yolo_class_id"])

    # emit XML
    lines = []
    lines.append("<View>")
    lines.append('  <Image name="image" value="$image"/>')
    lines.append('  <RectangleLabels name="label" toName="image">')

    for i, c in enumerate(high):
        name = c["coco_name"].replace("_", " ")
        color = HIGH_COLORS[i % len(HIGH_COLORS)]
        lines.append(f'    <Label value="{escape(name)}" background="{color}"/>')

    lines.append("    <!-- Medium priority -->")
    for i, c in enumerate(medium):
        name = c["coco_name"].replace("_", " ")
        color = MED_COLORS[i % len(MED_COLORS)]
        lines.append(f'    <Label value="{escape(name)}" background="{color}"/>')

    lines.append("    <!-- Low priority -->")
    for i, c in enumerate(low):
        name = c["coco_name"].replace("_", " ")
        color = LOW_COLORS[i % len(LOW_COLORS)]
        lines.append(f'    <Label value="{escape(name)}" background="{color}"/>')

    lines.append("  </RectangleLabels>")
    lines.append("</View>")

    xml = "\n".join(lines)

    with open(OUTPUT_XML_PATH, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"Saved Label Studio config to: {OUTPUT_XML_PATH}")
    print(xml)

    print("\n---")
    print(f"High: {len(high)}")
    print(f"Medium: {len(medium)}")
    print(f"Low: {len(low)}")
    print(f"Total: {len(high)+len(medium)+len(low)}")

if __name__ == "__main__":
    build()