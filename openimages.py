import fiftyone.zoo as foz

classes = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections"],
    max_samples=1
).default_classes

tablet_related = [
    c for c in classes
    if any(
        x in c.lower()
        for x in [
            "tablet",
            "phone",
            "computer",
            "monitor",
            "screen",
            "laptop"
        ]
    )
]

print(sorted(tablet_related))