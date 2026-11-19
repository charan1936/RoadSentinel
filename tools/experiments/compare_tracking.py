
from pathlib import Path
import csv
import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL = Path(
    r"D:\RoadSentinel-Datasets\training-runs\idd_yolov8n_full\weights\best.pt"
)
VIDEO = PROJECT_ROOT / "samples" / "road_video.mp4"
OUT = PROJECT_ROOT / "outputs" / "tracking_comparison"

OUT.mkdir(parents=True, exist_ok=True)

# Compare the current inference confidence against a stricter setting.
EXPERIMENTS = [
    {"name": "baseline_conf020", "tracker_conf": 0.20},
    {"name": "higher_conf035", "tracker_conf": 0.35},
]

def run_experiment(model, name, conf):
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_out = OUT / f"{name}.mp4"
    csv_out = OUT / f"{name}.csv"

    writer = cv2.VideoWriter(
        str(video_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create: {video_out}")

    counts = {}
    classes_by_track = {}
    records = 0
    frame_number = 0

    try:
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            out_csv = csv.writer(f)
            out_csv.writerow([
                "frame", "timestamp_seconds", "track_id",
                "class_name", "confidence",
                "x1", "y1", "x2", "y2"
            ])

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                result = model.track(
                    source=frame,
                    persist=True,
                    tracker="botsort.yaml",
                    imgsz=640,
                    conf=conf,
                    device=0,
                    verbose=False,
                )[0]

                writer.write(result.plot())

                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        score = float(box.conf[0].item())
                        cls_id = int(box.cls[0].item())
                        cls_name = result.names[cls_id]
                        tid = (
                            int(box.id[0].item())
                            if box.id is not None else None
                        )

                        if tid is not None:
                            counts[tid] = counts.get(tid, 0) + 1
                            classes_by_track.setdefault(tid, set()).add(
                                cls_name
                            )

                        x1, y1, x2, y2 = map(
                            int, box.xyxy[0].tolist()
                        )
                        out_csv.writerow([
                            frame_number,
                            round(frame_number / fps, 3),
                            tid if tid is not None else "",
                            cls_name,
                            round(score, 4),
                            x1, y1, x2, y2,
                        ])
                        records += 1

                frame_number += 1
                if frame_number % 200 == 0:
                    print(f"{name}: {frame_number}/{total_frames}")

    finally:
        cap.release()
        writer.release()

    multi_class = sum(
        len(classes) > 1 for classes in classes_by_track.values()
    )
    single_frame = sum(
        count == 1 for count in counts.values()
    )

    return {
        "experiment": name,
        "confidence": conf,
        "frames_processed": frame_number,
        "detection_records": records,
        "unique_track_ids": len(counts),
        "single_frame_tracks": single_frame,
        "multi_class_tracks": multi_class,
        "video": str(video_out),
        "csv": str(csv_out),
    }


def main():
    if not MODEL.is_file():
        raise FileNotFoundError(f"Model missing: {MODEL}")
    if not VIDEO.is_file() or VIDEO.stat().st_size == 0:
        raise FileNotFoundError(f"Video missing or empty: {VIDEO}")

    results = []
    for experiment in EXPERIMENTS:
        print(f"\nStarting {experiment['name']}")
        # A new model instance resets tracker state between experiments.
        model = YOLO(str(MODEL))
        results.append(run_experiment(
            model,
            experiment["name"],
            experiment["tracker_conf"],
        ))

    report = OUT / "comparison_summary.csv"
    with open(report, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nComparison complete.")
    print("Summary:", report)
    for row in results:
        print(
            f"{row['experiment']}: "
            f"{row['detection_records']} detections, "
            f"{row['unique_track_ids']} track IDs, "
            f"{row['multi_class_tracks']} multi-class tracks"
        )

if __name__ == "__main__":
    main()
