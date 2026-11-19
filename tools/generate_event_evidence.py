
from pathlib import Path
import csv
import cv2

ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = ROOT / "samples" / "road_video.mp4"
EVENTS_CSV = ROOT / "outputs" / "events" / "roadsentinel_events.csv"
DETECTIONS_CSV = ROOT / "outputs" / "roadsentinel_detections.csv"

EVIDENCE_DIR = ROOT / "outputs" / "events" / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
if not EVENTS_CSV.exists():
    raise FileNotFoundError(f"Events CSV not found: {EVENTS_CSV}")
if not DETECTIONS_CSV.exists():
    raise FileNotFoundError(f"Detections CSV not found: {DETECTIONS_CSV}")

# Index detections by frame and track ID so evidence can show
# the observed bounding box rather than only the raw video frame.
detections_by_key = {}

with DETECTIONS_CSV.open(
    "r", newline="", encoding="utf-8-sig"
) as file:
    for row in csv.DictReader(file):
        try:
            key = (int(row["frame"]), int(float(row["track_id"])))
            detections_by_key[key] = row
        except (ValueError, KeyError):
            continue

with EVENTS_CSV.open(
    "r", newline="", encoding="utf-8-sig"
) as file:
    events = list(csv.DictReader(file))

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
evidence_rows = []
created = 0
missing_frames = 0

try:
    for event in events:
        try:
            frame_number = int(event["evidence_frame"])
            track_id = int(event["track_id"])
        except (ValueError, KeyError):
            continue

        # Event and detection CSV frame numbers are zero-indexed, like
        # OpenCV's CAP_PROP_POS_FRAMES positions.
        if frame_number < 0 or frame_number >= frame_count:
            missing_frames += 1
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()

        if not ok:
            missing_frames += 1
            continue

        detection = detections_by_key.get((frame_number, track_id))

        if detection:
            try:
                x1 = int(float(detection["x1"]))
                y1 = int(float(detection["y1"]))
                x2 = int(float(detection["x2"]))
                y2 = int(float(detection["y2"]))

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), (0, 220, 100), 3
                )

                label = (
                    f"ID {track_id} "
                    f"{detection['class_name']} "
                    f"{float(detection['confidence']):.2f}"
                )
                cv2.putText(
                    frame,
                    label,
                    (max(10, x1), max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 220, 100),
                    2,
                    cv2.LINE_AA
                )
            except (ValueError, KeyError):
                pass

        event_label = (
            f"{event.get('event_type', 'event')} | "
            f"{event.get('start_seconds', '')} sec | "
            f"REVIEW REQUIRED"
        )
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (25, 25, 25), -1)
        cv2.putText(
            frame,
            event_label[:110],
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        safe_id = "".join(
            char if char.isalnum() or char in "-_" else "_"
            for char in event.get("event_id", "event")
        )
        image_path = EVIDENCE_DIR / f"{safe_id}.jpg"

        if cv2.imwrite(str(image_path), frame):
            created += 1
            evidence_rows.append({
                **event,
                "evidence_image": str(image_path),
                "evidence_generated": "yes",
            })
        else:
            missing_frames += 1
            evidence_rows.append({
                **event,
                "evidence_image": "",
                "evidence_generated": "no",
            })

finally:
    cap.release()

# Save a new file; the original event report remains untouched.
output_csv = EVENTS_CSV.with_name("roadsentinel_events_with_evidence.csv")

fieldnames = list(events[0].keys()) if events else []
for name in ("evidence_image", "evidence_generated"):
    if name not in fieldnames:
        fieldnames.append(name)

with output_csv.open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(evidence_rows)

print("\nEvidence generation finished.")
print(f"Events processed: {len(events)}")
print(f"Evidence images created: {created}")
print(f"Frames/images needing review: {missing_frames}")
print(f"Evidence folder: {EVIDENCE_DIR}")
print(f"Updated event report: {output_csv}")
