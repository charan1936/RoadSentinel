
from pathlib import Path
from collections import Counter

DATASET = Path(r"D:\RoadSentinel-Datasets\raw\IDD_Detection")

for split in ("train", "val", "test"):
    split_file = DATASET / f"{split}.txt"

    if not split_file.exists():
        print(f"\n{split.upper()}: SPLIT FILE MISSING")
        continue

    entries = [
        line.strip()
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    image_missing = []
    annotation_missing = []
    image_extensions = Counter()

    for item in entries:
        relative = Path(item)

        image_base = DATASET / "JPEGImages" / relative
        annotation = (DATASET / "Annotations" / relative).with_suffix(".xml")

        # IDD split entries omit extensions; check common image formats.
        candidates = [
            image_base.with_suffix(ext)
            for ext in (".jpg", ".jpeg", ".png")
        ]
        image = next((p for p in candidates if p.is_file()), None)

        if image is None:
            image_missing.append(item)
        else:
            image_extensions[image.suffix.lower()] += 1

        if not annotation.is_file():
            annotation_missing.append(item)

    print(f"\n{'=' * 45}")
    print(f"{split.upper()} SPLIT")
    print(f"Entries: {len(entries)}")
    print(f"Images found: {len(entries) - len(image_missing)}")
    print(f"Images missing: {len(image_missing)}")
    print(f"Annotations found: {len(entries) - len(annotation_missing)}")
    print(f"Annotations missing: {len(annotation_missing)}")
    print(f"Image extensions: {dict(image_extensions)}")

    if image_missing:
        print("First missing images:", image_missing[:5])

    if annotation_missing:
        print("First missing annotations:", annotation_missing[:5])

print("\nDataset audit finished. No files were modified.")
