"""Parking-zone candidate analysis for RoadSentinel.

This module identifies vehicle tracks that appear stationary inside a user-defined
ROI polygon. Results are review-only candidates; this does not determine whether
parking is legally prohibited.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any

VEHICLE_CLASSES = {
    "car", "motorcycle", "autorickshaw", "truck", "bus", "bicycle",
    "vehicle", "vehicle fallback", "trailer", "caravan", "train",
}

# IDD detection CSV track class names are generally vehicle classes.
MIN_RELIABLE_CONFIDENCE = 0.25
MIN_OBSERVATIONS = 3
MAX_TRACK_GAP_SECONDS = 3.0
# Pixel movement is normalized by frame diagonal, so this threshold is
# resolution-independent. 0.012 ~= 1.2% of the frame diagonal per second.
MAX_CENTER_SPEED_DIAGONAL_PER_SECOND = 0.012
MAX_CENTER_SPREAD_DIAGONAL = 0.025


def _point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _get(track: dict[str, Any], key: str, default=None):
    return track.get(key, default) if isinstance(track, dict) else default


def _normalize_track_points(points: list[dict[str, Any]]) -> list[dict[str, float]]:
    normalized = []
    for point in points:
        try:
            frame = int(point.get("frame", point.get("frame_number", 0)))
            timestamp = float(point.get("timestamp_seconds", point.get("timestamp", 0.0)))
            x1, y1, x2, y2 = (float(point[k]) for k in ("x1", "y1", "x2", "y2"))
            confidence = float(point.get("confidence", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (timestamp, x1, y1, x2, y2, confidence)):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        # Use bottom-center for zone membership: it approximates the vehicle's
        # contact point with the road better than the box center.
        normalized.append({
            "frame": frame,
            "timestamp_seconds": timestamp,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": confidence,
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
            "foot_x": (x1 + x2) / 2.0,
            "foot_y": y2,
        })
    normalized.sort(key=lambda p: (p["timestamp_seconds"], p["frame"]))
    return normalized


def _stationary_metrics(points: list[dict[str, float]], width: int, height: int):
    if len(points) < MIN_OBSERVATIONS:
        return None

    times = [p["timestamp_seconds"] for p in points]
    duration = max(times) - min(times)
    if duration <= 0:
        return None

    diagonal = math.hypot(width, height)
    if diagonal <= 0:
        return None

    centers = [(p["cx"], p["cy"]) for p in points]
    median_x = statistics.median(x for x, _ in centers)
    median_y = statistics.median(y for _, y in centers)
    radial_distances = [math.hypot(x - median_x, y - median_y) for x, y in centers]
    spread = statistics.median(radial_distances) / diagonal

    first, last = points[0], points[-1]
    net_speed = math.hypot(last["cx"] - first["cx"], last["cy"] - first["cy"]) / duration
    normalized_speed = net_speed / diagonal

    gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
    max_gap = max(gaps) if gaps else 0.0
    return {
        "duration": duration,
        "normalized_speed": normalized_speed,
        "normalized_spread": spread,
        "max_gap_seconds": max_gap,
        "median_x": median_x,
        "median_y": median_y,
    }


def analyze_parking_zone_tracks(
    tracks: dict[Any, list[dict[str, Any]]] | list[dict[str, Any]],
    roi_polygon: list[dict[str, float]],
    width: int,
    height: int,
    min_stationary_seconds: float = 10.0,
) -> list[dict[str, Any]]:
    """Return parking-zone candidate events from tracked detections.

    Args:
        tracks: Mapping track_id -> list of detection rows, as returned by
            roadsentinel_candidate_event_analyzer.load_tracks().
        roi_polygon: Polygon points with normalized x/y in [0, 1].
        width, height: Original video frame dimensions.
        min_stationary_seconds: Minimum time in the zone.

    Returns:
        Event dictionaries compatible with the RoadSentinel API.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Frame width and height must be positive.")
    if not isinstance(roi_polygon, list) or len(roi_polygon) < 3:
        raise ValueError("Parking ROI must contain at least three points.")

    polygon = []
    for point in roi_polygon:
        try:
            x, y = float(point["x"]), float(point["y"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("ROI points must contain numeric x and y values.")
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("ROI coordinates must be normalized to [0, 1].")
        polygon.append((x * width, y * height))

    if isinstance(tracks, list):
        grouped = defaultdict(list)
        for row in tracks:
            if isinstance(row, dict):
                grouped[row.get("track_id", row.get("id", "unknown"))].append(row)
        tracks = grouped

    if not isinstance(tracks, dict):
        raise ValueError("Tracks must be a mapping or list of detection rows.")

    events = []
    for raw_track_id, raw_points in tracks.items():
        if not isinstance(raw_points, list):
            continue
        points = _normalize_track_points(raw_points)
        if len(points) < MIN_OBSERVATIONS:
            continue

        class_names = [
            str(p.get("class_name", "unknown")).strip().lower()
            for p in raw_points if isinstance(p, dict)
        ]
        class_name = max(set(class_names), key=class_names.count) if class_names else "unknown"
        if class_name not in VEHICLE_CLASSES:
            continue

        # Keep only points whose vehicle bottom-center is inside the polygon.
        inside_points = [
            p for p in points
            if _point_in_polygon(p["foot_x"], p["foot_y"], polygon)
        ]
        if len(inside_points) < MIN_OBSERVATIONS:
            continue

        # Split by long gaps so a track disappearing and reappearing does not
        # falsely count as continuously stationary.
        segments = []
        current = [inside_points[0]]
        for point in inside_points[1:]:
            gap = point["timestamp_seconds"] - current[-1]["timestamp_seconds"]
            if gap > MAX_TRACK_GAP_SECONDS:
                segments.append(current)
                current = [point]
            else:
                current.append(point)
        segments.append(current)

        for segment_index, segment in enumerate(segments):
            if len(segment) < MIN_OBSERVATIONS:
                continue
            reliable = [p for p in segment if p["confidence"] >= MIN_RELIABLE_CONFIDENCE]
            if len(reliable) < MIN_OBSERVATIONS:
                continue

            metrics = _stationary_metrics(reliable, width, height)
            if not metrics or metrics["duration"] < float(min_stationary_seconds):
                continue
            if metrics["normalized_speed"] > MAX_CENTER_SPEED_DIAGONAL_PER_SECOND:
                continue
            if metrics["normalized_spread"] > MAX_CENTER_SPREAD_DIAGONAL:
                continue

            # Choose the observation nearest the median center for evidence.
            evidence_point = min(
                reliable,
                key=lambda p: math.hypot(p["cx"] - metrics["median_x"], p["cy"] - metrics["median_y"]),
            )
            mean_confidence = statistics.mean(p["confidence"] for p in reliable)
            track_id = str(raw_track_id)
            event_id = f"parking_{track_id}_{int(segment[0]['frame'])}_{segment_index}"

            notes = [
                "Vehicle track appears stationary within the user-calibrated polygon.",
                "This is not proof of illegal parking; verify signs, road rules, and context.",
            ]
            if metrics["max_gap_seconds"] > 1.0:
                notes.append(
                    f"Track observations contain a {metrics['max_gap_seconds']:.1f}s gap; "
                    "continuous presence is uncertain."
                )

            events.append({
                "event_id": event_id,
                "event_type": "parking_zone_candidate",
                "status": "needs_human_review",
                "track_id": track_id,
                "class_name": class_name,
                "timestamp_seconds": evidence_point["timestamp_seconds"],
                "start_seconds": segment[0]["timestamp_seconds"],
                "end_seconds": segment[-1]["timestamp_seconds"],
                "duration_seconds": metrics["duration"],
                "frame_number": evidence_point["frame"],
                "evidence_frame": evidence_point["frame"],
                "evidence_bbox": [
                    evidence_point["x1"], evidence_point["y1"],
                    evidence_point["x2"], evidence_point["y2"],
                ],
                "confidence": mean_confidence,
                "mean_detection_confidence": mean_confidence,
                "observations": len(reliable),
                "estimated_net_center_speed_px_per_second": (
                    metrics["normalized_speed"] * math.hypot(width, height)
                ),
                "description": (
                    f"{class_name} track {track_id} appears stationary inside the "
                    f"calibrated zone for {metrics['duration']:.1f} seconds."
                ),
                "notes": notes,
            })

    events.sort(key=lambda e: (e["start_seconds"], e["track_id"]))
    return events
