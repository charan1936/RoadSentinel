
from pathlib import Path
import cv2
from ultralytics import YOLO

MODEL_PATH = Path(
    r"D:\RoadSentinel-Datasets\training-runs\idd_yolov8n_full\weights\best.pt"
)

# Put a real road video in this folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEO_PATH = PROJECT_ROOT / "samples" / "road_video.mp4"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "roadsentinel_annotated.mp4"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

if not VIDEO_PATH.is_file() or VIDEO_PATH.stat().st_size == 0:
    raise FileNotFoundError(
        f"Valid video not found: {VIDEO_PATH}\n"
        "Place a real road video at this path and try again."
    )

model = YOLO(str(MODEL_PATH))

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError("OpenCV could not open the video.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0 or width <= 0 or height <= 0:
    cap.release()
    raise RuntimeError("The video has invalid FPS or dimensions.")

# MP4 output using H.264-compatible OpenCV codec fallback.
writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create the output video.")

print("Model:", MODEL_PATH)
print(f"Video: {width}x{height} at {fps:.2f} FPS")
print("Frames:", total_frames)
print("Running detection...")

frame_number = 0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(
            source=frame,
            imgsz=640,
            conf=0.25,
            device=0,
            verbose=False,
        )[0]

        annotated = result.plot()
        writer.write(annotated)

        frame_number += 1
        if frame_number % 100 == 0:
            print(f"Processed {frame_number}/{total_frames} frames")

finally:
    cap.release()
    writer.release()

print("\nDetection complete.")
print("Processed frames:", frame_number)
print("Output video:", OUTPUT_PATH)
