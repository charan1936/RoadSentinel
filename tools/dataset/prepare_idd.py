
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET
import shutil
import sys

SOURCE = Path(r"D:\RoadSentinel-Datasets\raw\IDD_Detection")
OUTPUT = Path(r"D:\RoadSentinel-Datasets\processed\IDD_YOLO")

# Keep one consistent mapping across training and validation.
CLASSES = [
    "motorcycle",
    "rider",
    "car",
    "person",
    "autorickshaw",
    "truck",
    "vehicle fallback",
    "bus",
    "traffic sign",
    "animal",
    "traffic light",
    "bicycle",
    "caravan",
    "train",
    "trailer",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def convert_split(split):
    split_file = SOURCE / f"{split}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing split file: {split_file}")

    image_out = OUTPUT / "images" / split
    label_out = OUTPUT / "labels" / split
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    entries = [
        line.strip()
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    errors = []
    class_counts = Counter()
    object_count = 0

    for index, entry in enumerate(entries, start=1):
        relative = Path(entry)
        image_src = (SOURCE / "JPEGImages" / relative).with_suffix(".jpg")
        xml_src = (SOURCE / "Annotations" / relative).with_suffix(".xml")

        if not image_src.is_file():
            errors.append(f"Missing image: {entry}")
            continue
        if not xml_src.is_file():
            errors.append(f"Missing annotation: {entry}")
            continue

        try:
            root = ET.parse(xml_src).getroot()
            size = root.find("size")
            width = int(size.findtext("width"))
            height = int(size.findtext("height"))
            if width <= 0 or height <= 0:
                raise ValueError("Invalid image dimensions")

            yolo_rows = []
            for obj in root.findall("object"):
                name = (obj.findtext("name") or "").strip()
                if name not in CLASS_TO_ID:
                    errors.append(f"Unknown class '{name}' in {entry}")
                    continue

                box = obj.find("bndbox")
                xmin = float(box.findtext("xmin"))
                ymin = float(box.findtext("ymin"))
                xmax = float(box.findtext("xmax"))
                ymax = float(box.findtext("ymax"))

                # Clip small annotation excursions to image boundaries.
                xmin = max(0.0, min(xmin, width))
                xmax = max(0.0, min(xmax, width))
                ymin = max(0.0, min(ymin, height))
                ymax = max(0.0, min(ymax, height))

                if xmax <= xmin or ymax <= ymin:
                    errors.append(f"Invalid bounding box in {entry}: {name}")
                    continue

                x_center = ((xmin + xmax) / 2) / width
                y_center = ((ymin + ymax) / 2) / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height

                yolo_rows.append(
                    f"{CLASS_TO_ID[name]} "
                    f"{x_center:.6f} {y_center:.6f} "
                    f"{box_width:.6f} {box_height:.6f}"
                )
                class_counts[name] += 1
                object_count += 1

            # Preserve the relative folder structure to avoid filename collisions.
            image_dst = (image_out / relative).with_suffix(".jpg")
            label_dst = (label_out / relative).with_suffix(".txt")
            image_dst.parent.mkdir(parents=True, exist_ok=True)
            label_dst.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(image_src, image_dst)
            label_dst.write_text(
                "\n".join(yolo_rows) + ("\n" if yolo_rows else ""),
                encoding="utf-8",
            )

        except Exception as exc:
            errors.append(f"Failed {entry}: {type(exc).__name__}: {exc}")

        if index % 2000 == 0:
            print(f"{split}: processed {index}/{len(entries)}")

    print(f"\n{split.upper()} SUMMARY")
    print(f"Entries: {len(entries)}")
    print(f"Objects converted: {object_count}")
    print(f"Errors: {len(errors)}")
    for name, count in class_counts.most_common():
        print(f"  {name}: {count}")

    if errors:
        error_file = OUTPUT / f"{split}_errors.txt"
        error_file.parent.mkdir(parents=True, exist_ok=True)
        error_file.write_text("\n".join(errors), encoding="utf-8")
        print(f"Error details saved to: {error_file}")


def write_yaml():
    yaml_path = OUTPUT / "idd.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    content = [
        f"path: {OUTPUT.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for class_id, name in enumerate(CLASSES):
        content.append(f"  {class_id}: {name!r}")

    yaml_path.write_text("\n".join(content) + "\n", encoding="utf-8")
    print(f"\nDataset YAML created: {yaml_path}")


if __name__ == "__main__":
    if not SOURCE.is_dir():
        sys.exit(f"Dataset folder not found: {SOURCE}")

    # Stop rather than accidentally mix new output with an old conversion.
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        sys.exit(
            f"Output already contains files: {OUTPUT}\n"
            "Inspect it before rerunning; do not overwrite blindly."
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    convert_split("train")
    convert_split("val")
    write_yaml()
    print("\nPreparation finished. Review error reports before training.")
