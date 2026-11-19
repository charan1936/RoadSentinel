
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

CSV_PATH = (
    ROOT / "experiments" / "tracker_comparison"
    / "confidence_025" / "botsort_conf025"
    / "botsort_conf025_detections.csv"
)

df = pd.read_csv(CSV_PATH)

def iou(a, b):
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    intersection = max(0, x2-x1) * max(0, y2-y1)

    area_a = max(0, a["x2"]-a["x1"]) * max(0, a["y2"]-a["y1"])
    area_b = max(0, b["x2"]-b["x1"]) * max(0, b["y2"]-b["y1"])

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0

duplicate_counts = []

for frame, group in df.groupby("frame"):
    rows = group.to_dict("records")

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if rows[i]["track_id"] == rows[j]["track_id"]:
                continue

            overlap = iou(rows[i], rows[j])

            if overlap >= 0.5:
                duplicate_counts.append({
                    "frame": frame,
                    "track_id_1": rows[i]["track_id"],
                    "track_id_2": rows[j]["track_id"],
                    "class_1": rows[i]["class_name"],
                    "class_2": rows[j]["class_name"],
                    "iou": overlap,
                })

result = pd.DataFrame(duplicate_counts)

output = ROOT / "experiments" / "tracker_comparison" / "confidence_025" / "overlapping_tracks.csv"

result.to_csv(output, index=False)

print("Overlapping different-track pairs:", len(result))
print("Frames affected:", result["frame"].nunique() if not result.empty else 0)
print("Saved:", output)

if not result.empty:
    print("\nClass pair counts:")
    print(
        result.groupby(["class_1", "class_2"]).size()
        .sort_values(ascending=False)
        .head(15).to_string()
    )
