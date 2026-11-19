"""Generate calibrated, review-only wrong-way movement candidates."""
from __future__ import annotations

import math
import statistics
from typing import Any


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if ((y1 > y) != (y2 > y)) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1:
            inside = not inside
        previous = current
    return inside


def analyze_wrong_way_tracks(
    tracks: dict[str, list[dict[str, Any]]],
    calibration: dict[str, Any],
    frame_width: int,
    frame_height: int,
    *,
    min_duration_seconds: float = 1.5,
    min_confidence: float = 0.35,
    min_displacement_pixels: float = 30.0,
    min_opposite_alignment: float = 0.65,
) -> list[dict[str, Any]]:
    """Compare movement inside a reviewer-drawn road polygon to its direction arrow.

    Coordinates in calibration are normalized to the frame. This reports only a
    candidate; a single image-space direction vector is sensitive to perspective.
    """
    polygon_values = calibration.get("roi_polygon", [])
    if len(polygon_values) < 3:
        raise ValueError("A road area must contain at least three points.")
    polygon = [(float(point["x"]) * frame_width, float(point["y"]) * frame_height) for point in polygon_values]
    start = calibration.get("direction_start", {})
    end = calibration.get("direction_end", {})
    dx = (float(end["x"]) - float(start["x"])) * frame_width
    dy = (float(end["y"]) - float(start["y"])) * frame_height
    direction_length = math.hypot(dx, dy)
    if direction_length < 1.0:
        raise ValueError("Expected direction arrow is too short.")
    direction = (dx / direction_length, dy / direction_length)

    events = []
    for track_id, points in tracks.items():
        reliable_inside = [
            point for point in points
            if point.get("confidence", 0.0) >= min_confidence
            and point_in_polygon(point["cx"], point["cy"], polygon)
        ]
        if len(reliable_inside) < 3:
            continue
        reliable_inside.sort(key=lambda point: (point["timestamp_seconds"], point["frame"]))
        first, last = reliable_inside[0], reliable_inside[-1]
        duration = max(0.0, last["timestamp_seconds"] - first["timestamp_seconds"])
        if duration < min_duration_seconds:
            continue

        window = max(1, len(reliable_inside) // 3)
        start_x = statistics.median(point["cx"] for point in reliable_inside[:window])
        start_y = statistics.median(point["cy"] for point in reliable_inside[:window])
        end_x = statistics.median(point["cx"] for point in reliable_inside[-window:])
        end_y = statistics.median(point["cy"] for point in reliable_inside[-window:])
        movement_x, movement_y = end_x - start_x, end_y - start_y
        displacement = math.hypot(movement_x, movement_y)
        if displacement < min_displacement_pixels:
            continue
        alignment = (movement_x * direction[0] + movement_y * direction[1]) / displacement
        if alignment > -min_opposite_alignment:
            continue

        mean_confidence = sum(point["confidence"] for point in reliable_inside) / len(reliable_inside)
        class_name = statistics.mode([point["class_name"] for point in reliable_inside])
        evidence_point = reliable_inside[len(reliable_inside) // 2]
        box_keys = ("x1", "y1", "x2", "y2")
        evidence_bbox = (
            [round(float(evidence_point[k]), 1) for k in box_keys]
            if all(k in evidence_point for k in box_keys) else []
        )
        events.append({
            "event_id": f"wrong_way_candidate_track_{track_id}",
            "event_type": "wrong_way_movement_candidate",
            "track_id": int(float(track_id)),
            "class_name": class_name,
            "timestamp_seconds": round(first["timestamp_seconds"], 3),
            "frame_number": first["frame"],
            "evidence_frame": evidence_point["frame"],
            "evidence_bbox": evidence_bbox,
            "evidence_image": "",
            "start_seconds": round(first["timestamp_seconds"], 3),
            "end_seconds": round(last["timestamp_seconds"], 3),
            "duration_seconds": round(duration, 3),
            "confidence": round(mean_confidence, 4),
            "mean_detection_confidence": round(mean_confidence, 4),
            "observations": len(reliable_inside),
            "projected_direction_alignment": round(alignment, 3),
            "opposite_direction_displacement_pixels": round(displacement, 2),
            "status": "needs_human_review",
            "description": "The track moved against the manually calibrated direction inside the selected road area.",
            "notes": (
                "Calibration, camera perspective, occlusion, or tracking errors can create false candidates. "
                "Verify the original video; this is not a confirmed traffic violation."
            ),
        })
    return sorted(events, key=lambda event: (event["start_seconds"], event["track_id"]))
