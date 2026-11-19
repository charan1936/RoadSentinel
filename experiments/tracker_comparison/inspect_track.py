
import cv2
import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[2]

video_path = root / "samples" / "road_video.mp4"
csv_path = root / "experiments" / "tracker_comparison" / "botsort" / "botsort_detections.csv"

output_dir = root / "experiments" / "tracker_comparison" / "track_4353_frames"
output_dir.mkdir(parents=True, exist_ok=True)

track_id = 4353

df = pd.read_csv(csv_path)
track = df[df["track_id"] == track_id]

if track.empty:
    raise ValueError(f"Track {track_id} not found.")

# Select frames covering the class transitions.
target_frames = [409, 434, 492, 532, 593, 627, 661]

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError("Could not open original video.")

for frame_number in target_frames:
    row = track[track["frame"] == frame_number]

    if row.empty:
        print(f"No detection for frame {frame_number}")
        continue

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()

    if not success:
        print(f"Could not read frame {frame_number}")
        continue

    detection = row.iloc[0]

    x1 = int(detection["x1"])
    y1 = int(detection["y1"])
    x2 = int(detection["x2"])
    y2 = int(detection["y2"])

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        4
    )

    label = (
        f"ID {track_id} | {detection['class_name']} | "
        f"Confidence {detection['confidence']:.2f} | "
        f"Frame {frame_number}"
    )

    cv2.putText(
        frame,
        label,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    output_path = output_dir / f"frame_{frame_number}.jpg"
    cv2.imwrite(str(output_path), frame)

    print("Saved:", output_path)

cap.release()

print("\nInspection frames created:", output_dir)
