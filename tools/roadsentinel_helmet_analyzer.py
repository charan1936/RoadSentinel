"""Review-only helmet candidates from an explicitly helmet-trained detector."""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Iterable


def _normalize_label(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())


def _label_role(value: Any) -> str | None:
    label = _normalize_label(value)
    if label in {"rider", "motorcyclist"}:
        return "rider"
    if label in {"without helmet", "no helmet", "nohelmet", "helmetless"}:
        return "no_helmet"
    if label in {"with helmet", "helmet", "helmeted"}:
        return "helmet"
    return None


def validate_helmet_model_classes(class_names: dict[Any, Any] | list[Any]) -> None:
    values = class_names.values() if isinstance(class_names, dict) else class_names
    roles = {_label_role(value) for value in values}
    required = {"rider", "no_helmet", "helmet"}
    missing = sorted(required - roles)
    if missing:
        raise ValueError(
            "Helmet checkpoint must expose explicit rider, without-helmet, and with-helmet classes; "
            f"missing roles: {', '.join(missing)}."
        )


def _valid_detection(row: dict[str, Any]) -> bool:
    try:
        values = [
            float(row[key])
            for key in ("timestamp_seconds", "confidence", "x1", "y1", "x2", "y2")
        ]
        int(row["frame"])
        int(float(row["track_id"]))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return (
        all(math.isfinite(value) for value in values)
        and 0.0 <= values[1] <= 1.0
        and values[4] > values[2]
        and values[5] > values[3]
    )


def _associate_status_box(
    status_box: dict[str, Any],
    riders: list[dict[str, Any]],
) -> dict[str, Any] | None:
    x1, y1, x2, y2 = (float(status_box[key]) for key in ("x1", "y1", "x2", "y2"))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    matches = []
    for rider in riders:
        rx1, ry1, rx2, ry2 = (float(rider[key]) for key in ("x1", "y1", "x2", "y2"))
        width, height = rx2 - rx1, ry2 - ry1
        if width <= 0 or height <= 0:
            continue
        relative_x = abs(center_x - (rx1 + rx2) / 2.0) / width
        relative_y = (center_y - ry1) / height
        if relative_x <= 0.4 and -0.05 <= relative_y <= 0.55:
            matches.append((relative_x + abs(relative_y - 0.15), rider))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    if len(matches) > 1 and matches[1][0] - matches[0][0] < 0.12:
        return None
    return matches[0][1]


def _intersection_over_smaller_box(
    first: dict[str, Any],
    second: dict[str, Any],
) -> float:
    intersection_width = max(
        0.0,
        min(float(first["x2"]), float(second["x2"]))
        - max(float(first["x1"]), float(second["x1"])),
    )
    intersection_height = max(
        0.0,
        min(float(first["y2"]), float(second["y2"]))
        - max(float(first["y1"]), float(second["y1"])),
    )
    intersection = intersection_width * intersection_height
    first_area = (float(first["x2"]) - float(first["x1"])) * (
        float(first["y2"]) - float(first["y1"])
    )
    second_area = (float(second["x2"]) - float(second["x1"])) * (
        float(second["y2"]) - float(second["y1"])
    )
    smaller_area = min(first_area, second_area)
    return intersection / smaller_area if smaller_area > 0 else 0.0


def _associate_primary_rider(
    specialist_rider: dict[str, Any],
    primary_riders: list[dict[str, Any]],
    *,
    min_overlap: float = 0.35,
) -> dict[str, Any] | None:
    matches = [
        (_intersection_over_smaller_box(specialist_rider, primary), primary)
        for primary in primary_riders
        if _intersection_over_smaller_box(specialist_rider, primary) >= min_overlap
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    if len(matches) > 1 and matches[0][0] - matches[1][0] < 0.08:
        return None
    return matches[0][1]


def analyze_helmet_candidates(
    detections: Iterable[dict[str, Any]],
    class_names: dict[Any, Any] | list[Any],
    *,
    primary_rider_detections: Iterable[dict[str, Any]] = (),
    min_detection_confidence: float = 0.65,
    min_rider_confidence: float = 0.25,
    min_observations: int = 3,
    min_duration_seconds: float = 0.5,
    min_no_helmet_ratio: float = 0.6,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Associate explicit helmet/no-helmet boxes to tracked riders over time.

    A no-helmet event requires repeated explicit ``without helmet`` model
    detections on one specialist-model rider track. A missing helmet detection
    is never interpreted as an unhelmeted rider.
    """
    validate_helmet_model_classes(class_names)
    if not math.isfinite(min_detection_confidence) or not 0.0 <= min_detection_confidence <= 1.0:
        raise ValueError("Helmet detection confidence must be in [0, 1].")
    if not math.isfinite(min_rider_confidence) or not 0.0 <= min_rider_confidence <= 1.0:
        raise ValueError("Rider detection confidence must be in [0, 1].")
    if min_observations < 2:
        raise ValueError("Helmet candidates require at least two observations.")
    if not math.isfinite(min_duration_seconds) or min_duration_seconds <= 0:
        raise ValueError("Helmet candidate duration must be finite and positive.")
    if not math.isfinite(min_no_helmet_ratio) or not 0.5 < min_no_helmet_ratio <= 1.0:
        raise ValueError("Minimum no-helmet observation ratio must be in (0.5, 1].")

    frames: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in detections:
        if not isinstance(row, dict) or not _valid_detection(row):
            continue
        row = dict(row)
        row["_role"] = _label_role(row.get("class_name", ""))
        if row["_role"]:
            frames[int(row["frame"])].append(row)

    primary_riders_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in primary_rider_detections:
        if not isinstance(row, dict) or not _valid_detection(row):
            continue
        if _label_role(row.get("class_name", "")) != "rider":
            continue
        if float(row["confidence"]) < min_rider_confidence:
            continue
        normalized = dict(row)
        normalized["track_id"] = str(int(float(row["track_id"])))
        primary_riders_by_frame[int(row["frame"])].append(normalized)

    observations: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    diagnostics = {
        "unmatched_status_detections": 0,
        "ambiguous_status_detections": 0,
        "uncorroborated_rider_detections": 0,
    }
    for frame_detections in frames.values():
        riders = [
            row for row in frame_detections
            if row["_role"] == "rider" and float(row["confidence"]) >= min_rider_confidence
        ]
        status_boxes = [
            row for row in frame_detections
            if row["_role"] in {"helmet", "no_helmet"}
            and float(row["confidence"]) >= min_detection_confidence
        ]
        frame_observations: dict[str, dict[str, Any]] = {}
        conflicting_tracks: set[str] = set()
        for status_box in status_boxes:
            rider = _associate_status_box(status_box, riders)
            if rider is None:
                diagnostics["unmatched_status_detections"] += 1
                continue
            primary_rider = _associate_primary_rider(
                rider,
                primary_riders_by_frame.get(int(status_box["frame"]), []),
            )
            if primary_rider is None:
                diagnostics["uncorroborated_rider_detections"] += 1
                continue
            rider_id = str(int(float(rider["track_id"])))
            previous = frame_observations.get(rider_id)
            if previous and previous["helmet_role"] != status_box["_role"]:
                conflicting_tracks.add(rider_id)
                continue
            if previous is None or float(status_box["confidence"]) > float(previous["helmet_detector_confidence"]):
                frame_observations[rider_id] = {
                    "frame": int(status_box["frame"]),
                    "timestamp_seconds": float(status_box["timestamp_seconds"]),
                    "rider_track_id": rider_id,
                    "primary_rider_track_id": str(int(float(primary_rider["track_id"]))),
                    "helmet_role": status_box["_role"],
                    "helmet_detector_confidence": float(status_box["confidence"]),
                    "rider_detector_confidence": float(rider["confidence"]),
                    "evidence_bbox": [float(status_box[key]) for key in ("x1", "y1", "x2", "y2")],
                    "rider_bbox": [float(rider[key]) for key in ("x1", "y1", "x2", "y2")],
                }
        diagnostics["ambiguous_status_detections"] += len(conflicting_tracks)
        for rider_id, observation in frame_observations.items():
            if rider_id not in conflicting_tracks:
                observations[rider_id][observation["frame"]] = observation

    events = []
    for rider_id, track_observations in observations.items():
        ordered = sorted(track_observations.values(), key=lambda item: (item["timestamp_seconds"], item["frame"]))
        primary_track_counts: dict[str, int] = defaultdict(int)
        for item in ordered:
            primary_track_counts[item["primary_rider_track_id"]] += 1
        primary_rider_id, corroborated_observations = max(
            primary_track_counts.items(),
            key=lambda item: (item[1], item[0]),
            default=("", 0),
        )
        if corroborated_observations < min_observations:
            continue
        ordered = [
            item for item in ordered
            if item["primary_rider_track_id"] == primary_rider_id
        ]
        no_helmet = [item for item in ordered if item["helmet_role"] == "no_helmet"]
        helmet = [item for item in ordered if item["helmet_role"] == "helmet"]
        if len(no_helmet) < min_observations:
            continue
        duration = no_helmet[-1]["timestamp_seconds"] - no_helmet[0]["timestamp_seconds"]
        if duration < min_duration_seconds:
            continue
        ratio = len(no_helmet) / len(ordered)
        if ratio < min_no_helmet_ratio:
            continue
        evidence = max(no_helmet, key=lambda item: (item["helmet_detector_confidence"], -item["frame"]))
        events.append({
            "event_id": f"helmetless_riding_candidate_track_{primary_rider_id}",
            "event_type": "helmetless_riding_candidate",
            "track_id": int(primary_rider_id),
            "helmet_model_track_id": int(rider_id),
            "track_source": "primary_rider_cross_checked",
            "class_name": "rider",
            "timestamp_seconds": round(evidence["timestamp_seconds"], 3),
            "frame_number": evidence["frame"],
            "evidence_frame": evidence["frame"],
            "evidence_bbox": [round(value, 1) for value in evidence["evidence_bbox"]],
            "start_seconds": round(no_helmet[0]["timestamp_seconds"], 3),
            "end_seconds": round(no_helmet[-1]["timestamp_seconds"], 3),
            "duration_seconds": round(duration, 3),
            "confidence": round(statistics.median(item["helmet_detector_confidence"] for item in no_helmet), 4),
            "helmet_detector_confidence": round(statistics.median(item["helmet_detector_confidence"] for item in no_helmet), 4),
            "rider_detector_confidence": round(statistics.median(item["rider_detector_confidence"] for item in no_helmet), 4),
            "observations": len(no_helmet),
            "helmet_observations": len(helmet),
            "status": "needs_human_review",
            "description": (
                "The helmet-specific model repeatedly classified the visible head region as without a helmet, "
                "and the primary road-scene detector independently identified the same person as a rider."
            ),
            "notes": (
                "Potential helmetless-riding candidate only. The helmet-model score is not a calibrated "
                "probability of a violation. Occlusion, image resolution, head-region association, and "
                "domain shift can cause errors; inspect the original frames before making a decision."
            ),
            "evidence_image": "",
        })
    return sorted(events, key=lambda event: (event["start_seconds"], event["track_id"])), diagnostics
