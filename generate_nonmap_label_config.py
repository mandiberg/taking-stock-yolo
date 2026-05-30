import os
from html import escape

from class_map_utils import get_id_to_name

# --- inputs ---
OUTPUT_XML_PATH = "nonmap_labelstudio_label_config.xml"

# Classes are loaded from config/custom_class_map.json
CLASSES = get_id_to_name(min_id=80, max_id=140)

# 40 visually distinct, equal-weight colors — varied hue, consistent saturation/brightness
FLAT_COLORS = [
    "#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4",
    "#42D4F4", "#F032E6", "#BFEF45", "#FABED4", "#469990",
    "#DCBEFF", "#9A6324", "#FFFAC8", "#800000", "#AAFFC3",
    "#808000", "#FFD8B1", "#000075", "#A9A9A9", "#FFFFFF",
    "#E53935", "#43A047", "#1E88E5", "#FB8C00", "#8E24AA",
    "#00ACC1", "#F4511E", "#6D4C41", "#00897B", "#3949AB",
    "#D81B60", "#039BE5", "#7CB342", "#FFB300", "#5E35B1",
    "#546E7A", "#C0CA33", "#E53935", "#00BCD4", "#FF7043",
    "#26A69A", "#EF9A9A", "#CE93D8", "#80DEEA", "#A5D6A7",
    "#FFF176", "#FFCC80",
]
HOTKEYS = {
    93: "d",
    123: "b",
    124: "t",
    127: "c",
    132: "v",
    133: "l",
    137: "p",

}

def build():
    all_classes = sorted(CLASSES.items())

    lines = []
    lines.append("<View>")
    lines.append('  <Image name="image" value="$image"/>')
    lines.append('  <RectangleLabels name="label" toName="image">')

    for i, (cid, name) in enumerate(all_classes):
        label = name
        # label = name.replace("_", " ")
        color = FLAT_COLORS[i % len(FLAT_COLORS)]
        if cid in HOTKEYS:
            line = f'    <Label value="{escape(label)}" background="{color}" hotkey="{HOTKEYS[cid]}"/>'
        else:
            line = f'    <Label value="{escape(label)}" background="{color}"/>'

        lines.append(line)

    lines.append("  </RectangleLabels>")
    lines.append("</View>")

    xml = "\n".join(lines)

    with open(OUTPUT_XML_PATH, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"Saved Label Studio config to: {OUTPUT_XML_PATH}")
    print(xml)

    print("\n---")
    print(f"Total: {len(all_classes)}")


if __name__ == "__main__":
    build()
