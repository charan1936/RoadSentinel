
"""RoadSentinel candidate-event analyzer.

Reads a RoadSentinel detections CSV and creates review-only event candidates.
It does NOT declare legal violations.

Detection continuity warnings are added to event notes when the gap between
consecutive detections exceeds the configured frame threshold.
They do not change event classification or reviewer status.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = {
    "frame", "timestamp_seconds", "track_id", "class_name", "confidence",
    "x1", "y1", "x2", "y2",
}

VEHICLE_CLASSES = {
    "car", "motorcycle", "autorickshaw", "truck", "bus", "bicycle",
    "vehicle", "vehicle fallback", "trailer", "caravan", "train",
}

# A gap is the difference between consecutive detected frame numbers.
# A gap greater than 15 frames triggers a warning, not rejection.
CONTINUITY_WARNING_GAP_FRAMES = 15

EVENT_OUTPUT_FIELDS = [
    "event_id", "event_type", "track_id", "class_name", "timestamp_seconds",
    "frame_number", "evidence_frame", "evidence_image", "start_seconds",
    "end_seconds", "duration_seconds", "confidence", "mean_detection_confidence",
    "observations", "estimated_net_center_speed_px_per_second",
    "description", "notes", "status", "related_track_ids", "participant_count",
    "image_space_separation_pixels", "track_source", "evidence_bbox",
    "helmet_detector_confidence", "rider_detector_confidence", "helmet_observations",
    "helmet_model_track_id", "vehicle_track_id", "vehicle_class_name",
    "person_displacement_pixels", "person_displacement_heights",
]


def number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value not in (None, "") else default
    except (ValueError, TypeError):
        return default


def load_tracks(csv_path: Path) -> dict[str, list[dict[str, Any]]]:
    tracks: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fields

        if missing:
            raise ValueError(
                "Detections CSV is missing columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            track_id = (row.get("track_id") or "").strip()

            if not track_id:
                continue

            try:
                numeric_track_id = int(float(track_id))
            except (ValueError, TypeError, OverflowError):
                continue

            if numeric_track_id < 0:
                continue

            track_id = str(numeric_track_id)

            x1, y1, x2, y2 = (
                number(row, key)
                for key in ("x1", "y1", "x2", "y2")
            )

            if x2 <= x1 or y2 <= y1:
                continue

            tracks[track_id].append({
                "frame": int(number(row, "frame")),
                "timestamp_seconds": number(row, "timestamp_seconds"),
                "class_name": (row.get("class_name") or "unknown").strip().lower(),
                "confidence": number(row, "confidence"),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "width": x2 - x1,
                "height": y2 - y1,
            })

    for points in tracks.values():
        points.sort(key=lambda p: (p["timestamp_seconds"], p["frame"]))

    return tracks


def get_continuity_warning(
    points: list[dict[str, Any]],
) -> str:
    """Return a warning for long intervals between consecutive detections."""

    frame_ordered = sorted(points, key=lambda p: p["frame"])

    largest_gap = 0
    largest_gap_seconds = 0.0

    for previous, current in zip(frame_ordered, frame_ordered[1:]):
        gap = current["frame"] - previous["frame"]

        if gap > largest_gap:
            largest_gap = gap
            largest_gap_seconds = max(
                0.0,
                current["timestamp_seconds"] - previous["timestamp_seconds"],
            )

    if largest_gap > CONTINUITY_WARNING_GAP_FRAMES:
        return (
            "Detection continuity warning: the largest interval between "
            f"consecutive detections was {largest_gap} frames "
            f"(approximately {largest_gap_seconds:.2f} seconds). "
            "Object movement or tracking identity may be uncertain during "
            "this interval. Review the video before drawing conclusions."
        )

    return ""


def analyze_track(
    track_id: str,
    points: list[dict[str, Any]],
    min_duration: float,
    stationary_px_per_second: float,
    min_confidence: float,
) -> list[dict[str, Any]]:

    if len(points) < 3:
        return []

    start, end = points[0], points[-1]
    duration = max(
        0.0,
        end["timestamp_seconds"] - start["timestamp_seconds"],
    )

    if duration < min_duration:
        return []

    reliable = [
        point for point in points
        if point["confidence"] >= min_confidence
    ]

    if len(reliable) < 3:
        return []

    xs = [point["cx"] for point in reliable]
    ys = [point["cy"] for point in reliable]

    mx, my = statistics.median(xs), statistics.median(ys)

    radial = [
        math.hypot(point["cx"] - mx, point["cy"] - my)
        for point in reliable
    ]

    spread_px = statistics.median(radial) if radial else 0.0

    first, last = reliable[0], reliable[-1]

    elapsed = max(
        0.001,
        last["timestamp_seconds"] - first["timestamp_seconds"],
    )

    net_speed_px_s = (
        math.hypot(
            last["cx"] - first["cx"],
            last["cy"] - first["cy"],
        ) / elapsed
    )

    class_name = statistics.mode(
        [point["class_name"] for point in reliable]
    )

    mean_conf = (
        sum(point["confidence"] for point in reliable) / len(reliable)
    )

    # Continuity warnings are informational. They do not change classification.
    continuity_warning = get_continuity_warning(points)

    stationary = (
        class_name in VEHICLE_CLASSES
        and net_speed_px_s <= stationary_px_per_second
        and spread_px <= stationary_px_per_second * 0.8
    )

    common_fields = {
        "track_id": track_id,
        "class_name": class_name,
        "timestamp_seconds": round(first["timestamp_seconds"], 3),
        "frame_number": first["frame"],
        "evidence_frame": first["frame"],
        "evidence_image": "",
        "start_seconds": round(first["timestamp_seconds"], 3),
        "end_seconds": round(last["timestamp_seconds"], 3),
        "duration_seconds": round(elapsed, 3),
        "confidence": round(mean_conf, 4),
        "mean_detection_confidence": round(mean_conf, 4),
        "observations": len(reliable),
        "estimated_net_center_speed_px_per_second": round(
            net_speed_px_s, 2
        ),
    }

    events: list[dict[str, Any]] = []

    if stationary:
        notes = (
            "This may reflect traffic, occlusion, or tracking noise; "
            "it is not proof of illegal parking or a violation."
        )

        if continuity_warning:
            notes += " " + continuity_warning

        events.append({
            **common_fields,
            "event_id": f"stationary_candidate_track_{track_id}",
            "event_type": "stationary_object_candidate",
            "status": "needs_human_review",
            "description": (
                "A vehicle track appears nearly stationary in image coordinates."
            ),
            "notes": notes,
        })

    else:
        notes = (
            "Camera calibration and road geometry are required before "
            "inferring direction or a traffic violation."
        )

        if continuity_warning:
            notes += " " + continuity_warning

        events.append({
            **common_fields,
            "event_id": f"movement_observation_track_{track_id}",
            "event_type": "movement_observation",
            "status": "observation",
            "description": "A tracked object was observed over time.",
            "notes": notes,
        })

    return events


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create review-only candidate events from RoadSentinel "
            "detections CSV."
        )
    )

    parser.add_argument(
        "detections_csv",
        type=Path,
        help="Path to *_detections.csv",
    )

    parser.add_argument(
        "--min-duration",
        type=float,
        default=3.0,
        help="Minimum track duration in seconds (default: 3)",
    )

    parser.add_argument(
        "--stationary-speed",
        type=float,
        default=5.0,
        help="Stationary heuristic threshold in pixels/second (default: 5)",
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.35,
        help="Minimum confidence for track summaries (default: 0.35)",
    )

    args = parser.parse_args()

    source = args.detections_csv.resolve()

    if not source.is_file():
        parser.error(f"CSV file not found: {source}")

    tracks = load_tracks(source)
    events: list[dict[str, Any]] = []

    for track_id, points in tracks.items():
        events.extend(
            analyze_track(
                track_id,
                points,
                args.min_duration,
                args.stationary_speed,
                args.min_confidence,
            )
        )

    events.sort(
        key=lambda event: (
            event["start_seconds"],
            event["track_id"],
            event["event_type"],
        )
    )

    output_csv = source.with_name(
        source.stem.replace("_detections", "") + "_candidate_events.csv"
    )

    output_json = source.with_name(
        source.stem.replace("_detections", "") + "_candidate_events.json"
    )

    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=EVENT_OUTPUT_FIELDS,
        )
        writer.writeheader()
        writer.writerows(events)

    payload = {
        "source_detections_csv": str(source),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tracked_objects_analyzed": len(tracks),
        "candidate_events": len(events),
        "event_type_counts": {
            event_type: sum(
                1 for event in events
                if event["event_type"] == event_type
            )
            for event_type in sorted({
                event["event_type"] for event in events
            })
        },
        "interpretation": (
            "These are heuristic observations for human review, not "
            "confirmed traffic violations. Pixel movement is not "
            "calibrated real-world speed."
        ),
        "events": events,
    }

    output_json.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(f"Tracked objects analyzed: {len(tracks)}")
    print(f"Candidate events created: {len(events)}")
    print(f"CSV:  {output_csv}")
    print(f"JSON: {output_json}")
    print("Reminder: review candidates manually; this script does not confirm violations.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
