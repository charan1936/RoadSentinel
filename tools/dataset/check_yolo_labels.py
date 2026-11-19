
from pathlib import Path
from collections import Counter
import math

ROOT = Path(r"D:\RoadSentinel-Datasets\processed\IDD_YOLO")
SPLITS = ("train", "val")
NUM_CLASSES = 15

for split in SPLITS:
    image_root = ROOT / "images" / split
    label_root = ROOT / "labels" / split

    images = list(image_root.rglob("*.jpg"))
    labels = list(label_root.rglob("*.txt"))

    print(f"\n{'=' * 40}")
    print(f"{split.upper()} LABEL AUDIT")
    print("Images:", len(images))
    print("Label files:", len(labels))

    errors = []
    class_counts = Counter()
    empty_labels = 0
    boxes = 0

    for label_file in labels:
        try:
            lines = label_file.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            errors.append(f"Unreadable label {label_file}: {exc}")
            continue

        if not lines:
            empty_labels += 1

        for line_number, line in enumerate(lines, start=1):
            parts = line.split()

            if len(parts) != 5:
                errors.append(
                    f"{label_file}:{line_number}: expected 5 fields"
                )
                continue

            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                errors.append(
                    f"{label_file}:{line_number}: non-numeric value"
                )
                continue

            if not 0 <= class_id < NUM_CLASSES:
                errors.append(
                    f"{label_file}:{line_number}: invalid class {class_id}"
                )
                continue

            if not all(map(math.isfinite, (x, y, w, h))):
                errors.append(
                    f"{label_file}:{line_number}: non-finite coordinate"
                )
                continue

            if not (0 <= x <= 1 and 0 <= y <= 1):
                errors.append(
                    f"{label_file}:{line_number}: center outside image"
                )
                continue

            if not (0 < w <= 1 and 0 < h <= 1):
                errors.append(
                    f"{label_file}:{line_number}: invalid width/height"
                )
                continue

            if x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6:
                errors.append(
                    f"{label_file}:{line_number}: box exceeds horizontal bounds"
                )
                continue

            if y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
                errors.append(
                    f"{label_file}:{line_number}: box exceeds vertical bounds"
                )
                continue

            class_counts[class_id] += 1
            boxes += 1

    # Check that every image has its corresponding label file.
    missing_labels = 0
    for image in images:
        relative = image.relative_to(image_root).with_suffix(".txt")
        if not (label_root / relative).is_file():
            missing_labels += 1

    print("Boxes checked:", boxes)
    print("Empty label files:", empty_labels)
    print("Images without label files:", missing_labels)
    print("Invalid label rows:", len(errors))

    if errors:
        for error in errors[:20]:
            print("ERROR:", error)
        report = ROOT / f"{split}_label_audit_errors.txt"
        report.write_text("\n".join(errors), encoding="utf-8")
        print("Full error report:", report)

    print("Class counts by ID:", dict(sorted(class_counts.items())))

print("\nAudit complete.")
