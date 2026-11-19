
from pathlib import Path

DATASET = Path(r"D:\RoadSentinel-Datasets\raw\IDD_Detection")

for split in ["train", "val", "test"]:
    split_file = DATASET / f"{split}.txt"

    if not split_file.exists():
        print(f"\n{split}: SPLIT FILE MISSING")
        continue

    entries = [
        line.strip()
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print(f"\n{split.upper()} entries: {len(entries)}")

    for item in entries[:5]:
        relative_path = Path(item)

        image_path = (
            DATASET / "JPEGImages" / relative_path
        ).with_suffix(".jpg")

        annotation_path = (
            DATASET / "Annotations" / relative_path
        ).with_suffix(".xml")

        print(
            f"{item} | "
            f"image: {image_path.exists()} | "
            f"annotation: {annotation_path.exists()}"
        )
