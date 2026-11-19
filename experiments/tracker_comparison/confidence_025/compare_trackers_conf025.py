
from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[3]
VIDEO_PATH = ROOT / "samples" / "road_video.mp4"

MODEL_PATH = Path(
    r"D:\RoadSentinel-Datasets\training-runs"
    r"\idd_yolov8n_full\weights\best.pt"
)

TRACKER_DIR = Path(
    r"C:\Projects\RoadSentinel\.venv\Lib\site-packages"
    r"\ultralytics\cfg\trackers"
)

OUTPUT_DIR = ROOT / "experiments" / "tracker_comparison" / "confidence_025"

CONFIDENCE = 0.25
IMAGE_SIZE = 640

TRACKERS = {
    "botsort_conf025": TRACKER_DIR / "botsort.yaml",
}
CSV_FIELDS = [
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
]


def validate_inputs():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    for name, config in TRACKERS.items():
        if not config.is_file():
            raise FileNotFoundError(
                f"Tracker configuration missing: {config}"
            )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is unavailable. Stopping to preserve "
            "identical GPU experiment conditions."
        )

    print("Input validation passed.")
    print("Video:", VIDEO_PATH)
    print("Model:", MODEL_PATH)
    print("GPU:", torch.cuda.get_device_name(0))


def run_tracker(name: str, tracker_config: Path):
    print("\n" + "=" * 60)
    print(f"RUNNING {name.upper()}")
    print("=" * 60)

    output_folder = OUTPUT_DIR / name
    output_folder.mkdir(parents=True, exist_ok=True)

    csv_path = output_folder / f"{name}_detections.csv"
    video_path = output_folder / f"{name}_annotated.mp4"
    summary_path = output_folder / f"{name}_summary.json"

    # Fresh model instance ensures tracker state is independent.
    model = YOLO(str(MODEL_PATH))

    device = 0
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Could not open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or expected_frames <= 0:
        raise RuntimeError("Invalid video metadata.")

    writer = None
    rows = []
    track_classes = defaultdict(Counter)
    track_frames = defaultdict(list)
    class_counts = Counter()
    processed_frames = 0

    started = time.perf_counter()

    results = model.track(
        source=str(VIDEO_PATH),
        stream=True,
        persist=True,
        tracker=str(tracker_config),
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        device=device,
        verbose=False,
    )

    try:
        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as handle:

            csv_writer = csv.DictWriter(
                handle,
                fieldnames=CSV_FIELDS,
            )
            csv_writer.writeheader()

            for result in results:
                frame_number = processed_frames
                timestamp = frame_number / fps

                frame = result.orig_img.copy()

                if writer is None:
                    writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )

                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Could not create {video_path}"
                        )

                boxes = result.boxes

                if boxes is not None and len(boxes):
                    coords = boxes.xyxy.cpu().tolist()
                    confidences = boxes.conf.cpu().tolist()
                    classes = boxes.cls.int().cpu().tolist()

                    ids = (
                        boxes.id.int().cpu().tolist()
                        if boxes.id is not None
                        else [None] * len(classes)
                    )

                    for box, confidence, class_id, track_id in zip(
                        coords,
                        confidences,
                        classes,
                        ids,
                    ):
                        class_name = str(
                            result.names.get(class_id, class_id)
                        )

                        x1, y1, x2, y2 = map(int, box)

                        x1 = max(0, min(x1, width - 1))
                        x2 = max(0, min(x2, width - 1))
                        y1 = max(0, min(y1, height - 1))
                        y2 = max(0, min(y2, height - 1))

                        if x2 <= x1 or y2 <= y1:
                            continue

                        if track_id is not None:
                            track_id = int(track_id)
                            track_classes[track_id][class_name] += 1
                            track_frames[track_id].append(frame_number)

                        class_counts[class_name] += 1

                        label = (
                            f"ID {track_id} {class_name} "
                            f"{confidence:.2f}"
                            if track_id is not None
                            else f"{class_name} {confidence:.2f}"
                        )

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (60, 210, 90),
                            2,
                        )

                        cv2.putText(
                            frame,
                            label,
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (60, 210, 90),
                            2,
                            cv2.LINE_AA,
                        )

                        csv_writer.writerow({
                            "frame": frame_number,
                            "timestamp_seconds": round(timestamp, 3),
                            "track_id": (
                                "" if track_id is None else track_id
                            ),
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": round(float(confidence), 5),
                            "x1": round(float(box[0]), 2),
                            "y1": round(float(box[1]), 2),
                            "x2": round(float(box[2]), 2),
                            "y2": round(float(box[3]), 2),
                        })

                writer.write(frame)
                processed_frames += 1

                if processed_frames % 100 == 0:
                    print(
                        f"{name}: {processed_frames}/"
                        f"{expected_frames} frames processed"
                    )

    finally:
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - started

    # Calculate statistics using tracked detections only.
    unique_tracks = len(track_frames)

    track_lengths = [
        len(frames) for frames in track_frames.values()
    ]

    class_inconsistent_tracks = sum(
        len(classes) > 1
        for classes in track_classes.values()
    )

    frame_gaps = []

    for frames in track_frames.values():
        ordered = sorted(set(frames))
        frame_gaps.extend(
            current - previous
            for previous, current in zip(
                ordered,
                ordered[1:],
            )
        )

    summary = {
        "tracker": name,
        "tracker_config": str(tracker_config),
        "model": str(MODEL_PATH),
        "video": str(VIDEO_PATH),
        "confidence": CONFIDENCE,
        "image_size": IMAGE_SIZE,
        "device": torch.cuda.get_device_name(0),
        "expected_frames": expected_frames,
        "processed_frames": processed_frames,
        "fps": fps,
        "duration_seconds": expected_frames / fps,
        "runtime_seconds": round(elapsed, 2),
        "processing_fps": round(
            processed_frames / elapsed, 2
        ) if elapsed else 0,
        "total_detections": sum(class_counts.values()),
        "unique_track_ids": unique_tracks,
        "class_inconsistent_tracks": class_inconsistent_tracks,
        "class_consistency_ratio": round(
            1 - class_inconsistent_tracks / unique_tracks, 4
        ) if unique_tracks else 0,
        "average_detections_per_track": round(
            sum(track_lengths) / unique_tracks, 2
        ) if unique_tracks else 0,
        "max_track_gap_frames": max(frame_gaps, default=0),
        "class_counts": dict(class_counts),
        "outputs": {
            "csv": str(csv_path),
            "video": str(video_path),
            "summary": str(summary_path),
        },
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nRESULT:", name.upper())
    print("Processed frames:", processed_frames)
    print("Unique tracks:", unique_tracks)
    print("Total detections:", summary["total_detections"])
    print("Class-inconsistent tracks:", class_inconsistent_tracks)
    print("Runtime:", round(elapsed, 2), "seconds")
    print("CSV:", csv_path)
    print("Video:", video_path)

    return summary


def main():
    validate_inputs()

    summaries = []

    for name, config in TRACKERS.items():
        summaries.append(run_tracker(name, config))

    comparison_path = OUTPUT_DIR / "comparison_summary.json"

    comparison_path.write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("TRACKER COMPARISON COMPLETE")
    print("=" * 60)

    for summary in summaries:
        print(
            summary["tracker"],
            "| Tracks:", summary["unique_track_ids"],
            "| Detections:", summary["total_detections"],
            "| Runtime:", summary["runtime_seconds"],
            "| Inconsistent tracks:",
            summary["class_inconsistent_tracks"],
        )

    print("\nComparison JSON:", comparison_path)


if __name__ == "__main__":
    main()
