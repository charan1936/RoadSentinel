
from pathlib import Path
import csv
import cv2
from ultralytics import YOLO

# ============================================================
# ROADSentinel CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = Path(
    r"D:\RoadSentinel-Datasets\training-runs\idd_yolov8n_full\weights\best.pt"
)

VIDEO_PATH = ROOT / "samples" / "road_video.mp4"

OUTPUT_DIR = ROOT / "outputs"

OUTPUT_VIDEO = OUTPUT_DIR / "roadsentinel_tracked.mp4"
OUTPUT_CSV = OUTPUT_DIR / "roadsentinel_detections.csv"

IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.40
DEVICE = 0

# ============================================================
# VALIDATE INPUTS
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Trained model not found:\n{MODEL_PATH}"
    )

if not VIDEO_PATH.is_file() or VIDEO_PATH.stat().st_size == 0:
    raise FileNotFoundError(
        f"Video not found or empty:\n{VIDEO_PATH}"
    )


# ============================================================
# DRAW CLEAN DETECTIONS
# ============================================================

def draw_clean_detections(frame, result):
    annotated = frame.copy()
    boxes = result.boxes
    names = result.names

    if boxes is None:
        return annotated

    height, width = frame.shape[:2]

    for box in boxes:
        confidence = float(box.conf[0].item())

        # Display filter only; does not improve model accuracy.
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        class_id = int(box.cls[0].item())
        class_name = names[class_id]

        track_id = None
        if box.id is not None:
            track_id = int(box.id[0].item())

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        # Green detection rectangle.
        color = (0, 210, 100)

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        if track_id is not None:
            label = f"#{track_id} {class_name} {confidence:.2f}"
        else:
            label = f"{class_name} {confidence:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.48
        thickness = 1

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )

        # Put the label above the box where possible.
        label_top = y1 - text_height - baseline - 6

        if label_top < 0:
            label_top = y1

        label_bottom = min(
            height - 1,
            label_top + text_height + baseline + 6,
        )

        label_right = min(
            width - 1,
            x1 + text_width + 8,
        )

        cv2.rectangle(
            annotated,
            (x1, label_top),
            (label_right, label_bottom),
            color,
            -1,
        )

        text_y = min(
            height - 1,
            label_top + text_height + 2,
        )

        cv2.putText(
            annotated,
            label,
            (x1 + 4, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    return annotated


# ============================================================
# MAIN VIDEO TRACKING PIPELINE
# ============================================================

def main():
    print("=" * 60)
    print("RoadSentinel - Object Tracking")
    print("=" * 60)

    print("Loading trained model...")
    model = YOLO(str(MODEL_PATH))

    print("Opening input video...")
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(
            "Input video has invalid FPS or dimensions."
        )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not create output video:\n{OUTPUT_VIDEO}"
        )

    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("Starting BoT-SORT tracking...")
    print()

    frame_number = 0
    total_detections = 0
    frames_with_detections = 0

    try:
        with open(
            OUTPUT_CSV,
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            csv_writer = csv.writer(csv_file)

            csv_writer.writerow([
                "frame",
                "timestamp_seconds",
                "track_id",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ])

            while True:
                success, frame = cap.read()

                if not success:
                    break

                # persist=True preserves tracker state between frames.
                result = model.track(
                    source=frame,
                    persist=True,
                    tracker="botsort.yaml",
                    imgsz=IMAGE_SIZE,
                    conf=0.20,
                    device=DEVICE,
                    verbose=False,
                )[0]

                annotated = draw_clean_detections(
                    frame,
                    result,
                )

                writer.write(annotated)

                boxes = result.boxes
                frame_detection_count = 0

                if boxes is not None:
                    for box in boxes:
                        confidence = float(box.conf[0].item())

                        if confidence < CONFIDENCE_THRESHOLD:
                            continue

                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0].tolist(),
                        )

                        class_id = int(box.cls[0].item())
                        class_name = result.names[class_id]

                        track_id = ""
                        if box.id is not None:
                            track_id = int(box.id[0].item())

                        timestamp = frame_number / fps

                        csv_writer.writerow([
                            frame_number,
                            round(timestamp, 3),
                            track_id,
                            class_id,
                            class_name,
                            round(confidence, 4),
                            x1,
                            y1,
                            x2,
                            y2,
                        ])

                        frame_detection_count += 1

                total_detections += frame_detection_count

                if frame_detection_count > 0:
                    frames_with_detections += 1

                frame_number += 1

                if frame_number % 100 == 0:
                    print(
                        f"Processed {frame_number}/{total_frames} "
                        f"frames | Current detections: "
                        f"{frame_detection_count}"
                    )

    finally:
        cap.release()
        writer.release()

    print()
    print("=" * 60)
    print("RoadSentinel tracking completed.")
    print("=" * 60)
    print(f"Frames processed: {frame_number}")
    print(f"Frames with detections: {frames_with_detections}")
    print(f"Detection records written: {total_detections}")
    print(f"Annotated video: {OUTPUT_VIDEO}")
    print(f"Detection CSV: {OUTPUT_CSV}")
    print()
    print("Note: Detection records are not confirmed violations.")


if __name__ == "__main__":
    main()
