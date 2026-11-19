
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

OUTPUT_DIR = (
    ROOT / "experiments" / "tracker_comparison"
    / "agnostic_nms"
)

TARGET_FRAMES = [409, 434, 492, 532, 593, 627, 661]

CONFIDENCE = 0.35
IMAGE_SIZE = 640
IOU_THRESHOLDS = [0.30, 0.50]


def calculate_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(
        0, box_a[3] - box_a[1]
    )

    area_b = max(0, box_b[2] - box_b[0]) * max(
        0, box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0


def get_detections(result, frame_number):
    detections = []

    if result.boxes is None:
        return detections

    for box, confidence, class_id in zip(
        result.boxes.xyxy.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
        result.boxes.cls.int().cpu().tolist(),
    ):
        detections.append({
            "frame": frame_number,
            "class_id": int(class_id),
            "class_name": str(result.names[int(class_id)]),
            "confidence": round(float(confidence), 5),
            "x1": round(float(box[0]), 2),
            "y1": round(float(box[1]), 2),
            "x2": round(float(box[2]), 2),
            "y2": round(float(box[3]), 2),
        })

    return detections


def count_overlaps(detections):
    counts = {threshold: 0 for threshold in IOU_THRESHOLDS}
    same_class = {threshold: 0 for threshold in IOU_THRESHOLDS}
    different_class = {threshold: 0 for threshold in IOU_THRESHOLDS}

    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            a = detections[i]
            b = detections[j]

            iou = calculate_iou(
                [a["x1"], a["y1"], a["x2"], a["y2"]],
                [b["x1"], b["y1"], b["x2"], b["y2"]],
            )

            for threshold in IOU_THRESHOLDS:
                if iou >= threshold:
                    counts[threshold] += 1

                    if a["class_id"] == b["class_id"]:
                        same_class[threshold] += 1
                    else:
                        different_class[threshold] += 1

    return {
        "total": counts,
        "same_class": same_class,
        "different_class": different_class,
    }


def save_annotated_frame(frame, detections, output_path):
    annotated = frame.copy()

    for detection in detections:
        x1 = int(detection["x1"])
        y1 = int(detection["y1"])
        x2 = int(detection["x2"])
        y2 = int(detection["y2"])

        label = (
            f'{detection["class_name"]} '
            f'{detection["confidence"]:.2f}'
        )

        cv2.rectangle(
            annotated, (x1, y1), (x2, y2),
            (0, 255, 0), 2
        )

        cv2.putText(
            annotated, label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 255, 0), 2
        )

    cv2.imwrite(str(output_path), annotated)


def main():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(VIDEO_PATH)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    print("GPU:", torch.cuda.get_device_name(0))

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    comparison = {}

    for frame_number in TARGET_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()

        if not success:
            print("Could not read frame:", frame_number)
            continue

        baseline_result = model.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device=0,
            agnostic_nms=False,
            verbose=False,
        )[0]

        agnostic_result = model.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device=0,
            agnostic_nms=True,
            verbose=False,
        )[0]

        baseline = get_detections(
            baseline_result, frame_number
        )

        agnostic = get_detections(
            agnostic_result, frame_number
        )

        baseline_overlaps = count_overlaps(baseline)
        agnostic_overlaps = count_overlaps(agnostic)

        comparison[str(frame_number)] = {
            "baseline": {
                "detection_count": len(baseline),
                "overlaps": baseline_overlaps,
                "detections": baseline,
            },
            "agnostic_nms": {
                "detection_count": len(agnostic),
                "overlaps": agnostic_overlaps,
                "detections": agnostic,
            },
        }

        save_annotated_frame(
            frame,
            baseline,
            OUTPUT_DIR / f"frame_{frame_number}_baseline.jpg",
        )

        save_annotated_frame(
            frame,
            agnostic,
            OUTPUT_DIR / f"frame_{frame_number}_agnostic.jpg",
        )

        print(
            f"Frame {frame_number}: "
            f"baseline={len(baseline)}, "
            f"agnostic={len(agnostic)}"
        )

    cap.release()

    output_json = OUTPUT_DIR / "nms_comparison.json"

    output_json.write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )

    print("\nExperiment completed.")
    print("Results:", OUTPUT_DIR)
    print("JSON:", output_json)


if __name__ == "__main__":
    main()
