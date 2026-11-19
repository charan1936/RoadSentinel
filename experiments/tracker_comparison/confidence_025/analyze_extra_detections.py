
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

BASELINE = ROOT / "experiments" / "tracker_comparison" / "botsort" / "botsort_detections.csv"

LOW_CONF = ROOT / "experiments" / "tracker_comparison" / "confidence_025" / "botsort_conf025" / "botsort_conf025_detections.csv"

OUTPUT = ROOT / "experiments" / "tracker_comparison" / "confidence_025"

base = pd.read_csv(BASELINE)
low = pd.read_csv(LOW_CONF)

def iou(a, b):
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, a["x2"] - a["x1"]) * max(0, a["y2"] - a["y1"])
    area_b = max(0, b["x2"] - b["x1"]) * max(0, b["y2"] - b["y1"])

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0

print("Matching detections by frame and bounding-box overlap...")

baseline_by_frame = {
    frame: group.to_dict("records")
    for frame, group in base.groupby("frame")
}

extra_rows = []

for row in low.to_dict("records"):
    candidates = baseline_by_frame.get(row["frame"], [])

    best_iou = max(
        (iou(row, candidate) for candidate in candidates),
        default=0
    )

    if best_iou < 0.5:
        extra_rows.append(row)

extras = pd.DataFrame(extra_rows)

output_path = OUTPUT / "extra_detections_conf025.csv"
extras.to_csv(output_path, index=False)

print("\nLow-confidence detections without baseline overlap:", len(extras))
print("Saved:", output_path)

if not extras.empty:
    print("\nConfidence distribution:")
    print(extras["confidence"].describe().round(3))

    print("\nClass distribution:")
    print(extras["class_name"].value_counts())

    print("\nSample detections:")
    print(
        extras[
            ["frame", "track_id", "class_name", "confidence",
             "x1", "y1", "x2", "y2"]
        ].head(20).to_string(index=False)
    )
