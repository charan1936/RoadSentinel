
from pathlib import Path
import csv
import json
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "outputs" / "roadsentinel_detections.csv"
OUTPUT_DIR = ROOT / "outputs" / "events"

EVENTS_CSV = OUTPUT_DIR / "roadsentinel_events.csv"
SUMMARY_JSON = OUTPUT_DIR / "roadsentinel_event_summary.json"

# Conservative starting settings; tune against reviewed footage.
MIN_OBSERVATIONS = 10
MIN_MOVEMENT_PIXELS = 20.0
STATIONARY_MOVEMENT_PIXELS = 12.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_detections(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {path}\n"
            "Check that your existing tracking script generated it."
        )

    detections = []

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {
            "frame", "timestamp_seconds", "track_id",
            "class_name", "confidence", "x1", "y1", "x2", "y2"
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            try:
                x1 = float(row["x1"])
                y1 = float(row["y1"])
                x2 = float(row["x2"])
                y2 = float(row["y2"])

                detections.append({
                    "frame": int(row["frame"]),
                    "timestamp": float(row["timestamp_seconds"]),
                    "track_id": int(float(row["track_id"])),
                    "class_name": row["class_name"].strip().lower(),
                    "confidence": float(row["confidence"]),
                    "cx": (x1 + x2) / 2,
                    "cy": (y1 + y2) / 2,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })
            except (ValueError, TypeError):
                continue

    return detections


def main():
    detections = read_detections(INPUT_CSV)

    # Only valid IDs can support track-based movement analysis.
    tracks = defaultdict(list)
    for item in detections:
        if item["track_id"] >= 0:
            tracks[item["track_id"]].append(item)

    events = []
    track_summary = []
    class_totals = Counter()

    for track_id, observations in tracks.items():
        observations.sort(key=lambda item: item["frame"])

        # One track can have inconsistent predicted classes.
        classes = Counter(item["class_name"] for item in observations)
        main_class, main_class_count = classes.most_common(1)[0]
        consistency = main_class_count / len(observations)

        first = observations[0]
        last = observations[-1]

        # Pixel displacement is a rough image-space measurement,
        # not real-world speed or distance.
        dx = last["cx"] - first["cx"]
        dy = last["cy"] - first["cy"]
        displacement = (dx * dx + dy * dy) ** 0.5

        duration = max(0.0, last["timestamp"] - first["timestamp"])
        stable_class = consistency >= 0.80

        track_summary.append({
            "track_id": track_id,
            "predominant_class": main_class,
            "observations": len(observations),
            "duration_seconds": round(duration, 3),
            "class_consistency": round(consistency, 3),
            "image_displacement_pixels": round(displacement, 2),
        })

        class_totals[main_class] += 1

        if len(observations) < MIN_OBSERVATIONS or not stable_class:
            continue

        # This is only a stationary-object candidate.
        # A fixed camera, perspective, occlusion and ID errors can
        # affect the result. It does not prove illegal parking.
        if (
            duration >= 2.0
            and displacement <= STATIONARY_MOVEMENT_PIXELS
        ):
            events.append({
                "event_id": f"RS-{track_id}-STOP",
                "event_type": "stationary_object_candidate",
                "track_id": track_id,
                "class_name": main_class,
                "start_seconds": round(first["timestamp"], 3),
                "end_seconds": round(last["timestamp"], 3),
                "duration_seconds": round(duration, 3),
                "confidence": round(
                    sum(x["confidence"] for x in observations)
                    / len(observations), 3
                ),
                "evidence_frame": first["frame"],
                "status": "needs_human_review",
                "notes": (
                    "Low image-space displacement only; "
                    "not proof of illegal parking or a traffic violation."
                ),
            })

        # A moving track is recorded as a movement observation,
        # not automatically as a violation.
        elif displacement >= MIN_MOVEMENT_PIXELS:
            events.append({
                "event_id": f"RS-{track_id}-MOVEMENT",
                "event_type": "movement_observation",
                "track_id": track_id,
                "class_name": main_class,
                "start_seconds": round(first["timestamp"], 3),
                "end_seconds": round(last["timestamp"], 3),
                "duration_seconds": round(duration, 3),
                "confidence": round(
                    sum(x["confidence"] for x in observations)
                    / len(observations), 3
                ),
                "evidence_frame": first["frame"],
                "status": "observation_not_violation",
                "notes": (
                    "Image-space movement detected. "
                    "Road direction and calibrated geometry are not available."
                ),
            })

    fields = [
        "event_id", "event_type", "track_id", "class_name",
        "start_seconds", "end_seconds", "duration_seconds",
        "confidence", "evidence_frame", "status", "notes"
    ]

    with EVENTS_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)

    summary = {
        "input_file": str(INPUT_CSV),
        "detection_rows_read": len(detections),
        "unique_valid_track_ids": len(tracks),
        "events_written": len(events),
        "event_types": dict(
            Counter(event["event_type"] for event in events)
        ),
        "track_class_counts": dict(class_totals),
        "important_limitations": [
            "Pixel movement is not calibrated real-world speed.",
            "Stationary candidates do not establish illegal parking.",
            "No helmet, triple-riding, wrong-way or red-light violations "
            "are confirmed by this script.",
            "Track IDs and predicted classes can be unstable.",
            "Every candidate requires review against the source video."
        ],
        "tracks": track_summary,
    }

    with SUMMARY_JSON.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nRoadSentinel Event Engine finished.")
    print(f"Detection rows read: {len(detections)}")
    print(f"Valid track IDs: {len(tracks)}")
    print(f"Candidate events: {len(events)}")
    print(f"Events CSV: {EVENTS_CSV}")
    print(f"Summary JSON: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
