
import pandas as pd
from itertools import combinations

files = {
    "BoT-SORT 30": "experiments/tracker_comparison/botsort/botsort_detections.csv",
    "BoT-SORT 15": "experiments/tracker_comparison/botsort_buffer15/botsort_buffer15_detections.csv",
    "BoT-SORT 10": "experiments/tracker_comparison/botsort_buffer10/botsort_buffer10_detections.csv",
}

def iou(a, b):
    left = max(a.x1, b.x1)
    top = max(a.y1, b.y1)
    right = min(a.x2, b.x2)
    bottom = min(a.y2, b.y2)

    intersection = max(0, right-left) * max(0, bottom-top)

    area_a = (a.x2-a.x1) * (a.y2-a.y1)
    area_b = (b.x2-b.x1) * (b.y2-b.y1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0

for name, path in files.items():
    df = pd.read_csv(path)
    overlap_pairs = 0
    high_overlap_pairs = 0
    same_id_pairs = 0

    for frame, group in df.groupby("frame"):
        detections = list(group.itertuples(index=False))

        for a, b in combinations(detections, 2):
            score = iou(a, b)

            if score >= 0.30:
                overlap_pairs += 1

            if score >= 0.50:
                high_overlap_pairs += 1

            if score >= 0.50 and a.track_id == b.track_id:
                same_id_pairs += 1

    print(f"\n{name}")
    print("Overlapping detection pairs (IoU >= 0.30):", overlap_pairs)
    print("High-overlap pairs (IoU >= 0.50):", high_overlap_pairs)
    print("Same-ID high-overlap pairs:", same_id_pairs)
