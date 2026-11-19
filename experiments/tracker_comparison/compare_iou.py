
import pandas as pd

original = pd.read_csv(
    "experiments/tracker_comparison/botsort/botsort_detections.csv"
)

buffer10 = pd.read_csv(
    "experiments/tracker_comparison/botsort_buffer10/botsort_buffer10_detections.csv"
)

frames = [409, 434, 492, 532, 593, 627, 661]
original_track_id = 4353


def calculate_iou(a, b):
    x_left = max(a.x1, b.x1)
    y_top = max(a.y1, b.y1)
    x_right = min(a.x2, b.x2)
    y_bottom = min(a.y2, b.y2)

    intersection = max(0, x_right - x_left) * max(
        0, y_bottom - y_top
    )

    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0


print("FRAME | ORIGINAL CLASS | NEW ID | NEW CLASS | IoU")
print("-" * 75)

for frame in frames:
    old = original[
        (original.frame == frame)
        & (original.track_id == original_track_id)
    ]

    new = buffer10[buffer10.frame == frame]

    if old.empty:
        print(frame, "| Original detection missing")
        continue

    if new.empty:
        print(frame, "| New detections missing")
        continue

    old_box = next(old.itertuples(index=False))

    best_detection = None
    best_iou = 0

    for detection in new.itertuples(index=False):
        score = calculate_iou(old_box, detection)

        if score > best_iou:
            best_iou = score
            best_detection = detection

    if best_detection is None:
        print(frame, "| No matching detection")
        continue

    print(
        f"{frame:5} | "
        f"{old_box.class_name:14} | "
        f"{int(best_detection.track_id):6} | "
        f"{best_detection.class_name:12} | "
        f"{best_iou:.3f}"
    )
