import os
from html import escape

# --- inputs ---
OUTPUT_XML_PATH = "nonmap_labelstudio_label_config.xml"

# Classes 80-119 from the taking-stock class map
CLASSES = {
    80: "Sign",
    81: "Gift",
    82: "Money",
    83: "Bag",
    84: "Valentine",
    85: "Salad",
    86: "Dumbbell",
    87: "Flag",
    88: "Groceries",
    89: "Chestpiece",
    90: "Stethoscope",
    91: "Gun",
    92: "Headphones",
    93: "Clipboard",
    94: "Piggybank",
    95: "Creditcard",
    96: "Bitcoin",
    97: "Rose",
    98: "Lily",
    99: "Iris",
    100: "Tulip",
    101: "Lisianthus",
    102: "Orchid",
    103: "Peony",
    104: "Sunflower",
    105: "Daisy",
    106: "Daffodil",
    107: "Hydrangea",
    108: "Pistol",
    109: "Rifle",
    110: "Mask",
    111: "Facial",
    112: "Sheetmask",
    113: "Eyepatch",
    114: "Sleepmask",
    115: "Masquerade_mask",
    116: "Cucumber",
    117: "Kiwi",
    118: "Lemon_slice",
    119: "Avocado_half",
    120: "Eyeglasses",
    121: "Cigarette",
    122: "Vape",
    123: "Boxing_gloves",
    124: "Tablet",
    125: "Picture_frame",
    126: "Playing_cards",
    127: "calculator",
    128: "megaphone",

}

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


def build():
    all_classes = sorted(CLASSES.items())

    lines = []
    lines.append("<View>")
    lines.append('  <Image name="image" value="$image"/>')
    lines.append('  <RectangleLabels name="label" toName="image">')

    for i, (cid, name) in enumerate(all_classes):
        label = name.replace("_", " ")
        color = FLAT_COLORS[i % len(FLAT_COLORS)]
        lines.append(f'    <Label value="{escape(label)}" background="{color}"/>')

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
