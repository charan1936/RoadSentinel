
from pathlib import Path
import json

import cv2
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = ROOT / "samples" / "road_video.mp4"

MODEL_PATH = Path(
    r"D:\RoadSentinel-Datasets\training-runs"
    r"\idd_yolov8n_full\weights\best.pt"
)

OUTPUT_DIR = ROOT / "experiments" / "tracker_comparison" / "detection_only"

TARGET_FRAMES = [409, 434, 492, 532, 593, 627, 661]

CONFIDENCE = 0.35
IMAGE_SIZE = 640


def main():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(VIDEO_PATH)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = YOLO(str(MODEL_PATH))

    print("GPU:", torch.cuda.get_device_name(0))

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    results_data = []

    for frame_number in TARGET_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()

        if not success:
            print("Could not read frame:", frame_number)
            continue

        # Detection-only inference: no tracking or persistent IDs.
        result = model.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device=0,
            verbose=False,
        )[0]

        annotated = frame.copy()
        frame_detections = []

        if result.boxes is not None and len(result.boxes):
            coords = result.boxes.xyxy.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()

            for box, confidence, class_id in zip(
                coords, confidences, classes
            ):
                x1, y1, x2, y2 = map(int, box)

                class_name = str(
                    result.names.get(class_id, class_id)
                )

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                label = f"{class_name} {confidence:.2f}"

                cv2.putText(
                    annotated,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

                frame_detections.append({
                    "frame": frame_number,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(float(confidence), 5),
                    "x1": round(float(box[0]), 2),
                    "y1": round(float(box[1]), 2),
                    "x2": round(float(box[2]), 2),
                    "y2": round(float(box[3]), 2),
                })

        output_image = OUTPUT_DIR / f"frame_{frame_number}.jpg"

        cv2.imwrite(str(output_image), annotated)

        results_data.extend(frame_detections)

        print(
            f"Frame {frame_number}: "
            f"{len(frame_detections)} detections"
        )

    cap.release()

    json_path = OUTPUT_DIR / "detection_results.json"

    json_path.write_text(
        json.dumps(results_data, indent=2),
        encoding="utf-8",
    )

    print("\nDetection-only validation completed.")
    print("Output folder:", OUTPUT_DIR)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
