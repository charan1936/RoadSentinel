
import json
import pandas as pd
from collections import defaultdict

RAW_PATH = "experiments/tracker_comparison/detection_only/detection_results.json"
TRACKED_PATH = "outputs/video_jobs/e925455d0c95_detections.csv"

FRAMES = [409, 434, 492, 532, 593, 627, 661]
IOU_THRESHOLD = 0.5

with open(RAW_PATH, "r") as f:
    raw = json.load(f)

tracked = pd.read_csv(TRACKED_PATH)
tracked = tracked[tracked["frame"].isin(FRAMES)]

raw_by_frame = defaultdict(list)

for detection in raw:
    raw_by_frame[detection["frame"]].append(detection)


def calculate_iou(a, b):
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])

    union = area_a + area_b - intersection

    return intersection / union if union else 0


matched = 0
class_mismatches = 0
unmatched = 0

print(
    "FRAME | TRACK ID | TRACK CLASS | RAW CLASS | IoU | RAW CONF"
)

for _, track in tracked.iterrows():
    frame = int(track["frame"])
    candidates = raw_by_frame[frame]

    if not candidates:
        unmatched += 1
        continue

    best_detection = max(
        candidates,
        key=lambda raw_detection: calculate_iou(
            track, raw_detection
        ),
    )

    score = calculate_iou(track, best_detection)

    if score < IOU_THRESHOLD:
        unmatched += 1
        continue

    matched += 1

    mismatch = (
        track["class_name"] != best_detection["class_name"]
    )

    if mismatch:
        class_mismatches += 1

    print(
        f"{frame:4} | "
        f"{int(track['track_id']):7} | "
        f"{track['class_name']:12} | "
        f"{best_detection['class_name']:12} | "
        f"{score:.3f} | "
        f"{best_detection['confidence']:.4f}"
    )

print("\nSUMMARY")
print("Matched detections:", matched)
print("Class mismatches:", class_mismatches)
print("Unmatched tracked detections:", unmatched)
print("IoU threshold:", IOU_THRESHOLD)
