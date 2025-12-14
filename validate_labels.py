from pathlib import Path

LABEL_DIRS = [
    "yolo_dataset/train/labels",
    "yolo_dataset/val/labels",
]

NC = 15  # your nc

bad = []

for label_dir in LABEL_DIRS:
    for f in Path(label_dir).rglob("*.txt"):
        with open(f) as fh:
            for i, line in enumerate(fh, 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    bad.append((f, i, "wrong columns"))
                    continue

                cls, x, y, w, h = parts
                cls = int(cls)
                x, y, w, h = map(float, (x, y, w, h))

                if not (0 <= cls < NC):
                    bad.append((f, i, "class out of range"))

                if w <= 0 or h <= 0:
                    bad.append((f, i, "zero or negative size"))

                if not all(0 <= v <= 1 for v in (x, y, w, h)):
                    bad.append((f, i, "coords out of [0,1]"))

print("BAD LABELS:", len(bad))
for b in bad[:20]:
    print(b)
