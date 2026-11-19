"""Conservative image-space interaction candidates from tracked detections."""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any

MOTORCYCLE_CLASSES = {"motorcycle", "motorbike"}
RIDER_CLASSES = {"rider"}
PERSON_CLASSES = {"person", "pedestrian"}
VEHICLE_CLASSES = {
    "car", "motorcycle", "motorbike", "autorickshaw", "truck", "bus",
    "vehicle", "vehicle fallback", "trailer", "caravan", "train",
}


def _class_name(points: list[dict[str, Any]]) -> str:
    names = [str(point.get("class_name", "unknown")).strip().lower() for point in points]
    return statistics.mode(names) if names else "unknown"


def _valid_point(point: dict[str, Any]) -> bool:
    try:
        values = [float(point[key]) for key in ("timestamp_seconds", "confidence", "x1", "y1", "x2", "y2")]
        int(point.get("frame", point.get("frame_number", 0)))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return all(math.isfinite(value) for value in values) and values[4] > values[2] and values[5] > values[3]


def _frame_index(
    tracks: dict[str, list[dict[str, Any]]],
    min_confidence: float,
) -> dict[int, dict[str, tuple[str, dict[str, Any]]]]:
    frames: dict[int, dict[str, tuple[str, dict[str, Any]]]] = defaultdict(dict)
    for track_id, points in tracks.items():
        for point in points:
            if not isinstance(point, dict) or not _valid_point(point) or float(point["confidence"]) < min_confidence:
                continue
            frame = int(point.get("frame", point.get("frame_number", 0)))
            frames[frame][str(track_id)] = (str(point.get("class_name", "unknown")).strip().lower(), point)
    return frames


def _overlap_over_smaller_box(first: dict[str, Any], second: dict[str, Any]) -> float:
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
    first_area = (float(first["x2"]) - float(first["x1"])) * (float(first["y2"]) - float(first["y1"]))
    second_area = (float(second["x2"]) - float(second["x1"])) * (float(second["y2"]) - float(second["y1"]))
    smaller_area = min(first_area, second_area)
    return intersection / smaller_area if smaller_area > 0 else 0.0


def _distinct_riders(
    riders: list[tuple[str, dict[str, Any]]],
    *,
    duplicate_overlap_threshold: float = 0.8,
) -> list[tuple[str, dict[str, Any]]]:
    """Suppress near-identical rider boxes that represent the same visible person."""
    distinct: list[tuple[str, dict[str, Any]]] = []
    for rider in sorted(riders, key=lambda item: float(item[1]["confidence"]), reverse=True):
        if any(_overlap_over_smaller_box(rider[1], existing[1]) >= duplicate_overlap_threshold for existing in distinct):
            continue
        distinct.append(rider)
    return distinct


def analyze_triple_riding_candidates(
    tracks: dict[str, list[dict[str, Any]]],
    *,
    min_riders: int = 3,
    min_candidate_frames: int = 5,
    min_candidate_duration_seconds: float = 0.75,
    min_confidence: float = 0.35,
) -> list[dict[str, Any]]:
    """Find sustained frames with three distinct, spatially associated rider tracks.

    Box association remains approximate and produces candidates only; source-frame
    review is required to verify that riders occupy the same motorcycle.
    """
    if min_riders < 3 or min_candidate_frames < 3:
        raise ValueError("Triple-riding thresholds must require at least three riders across three frames.")
    if not math.isfinite(min_candidate_duration_seconds) or min_candidate_duration_seconds <= 0:
        raise ValueError("Triple-riding minimum duration must be finite and positive.")
    if not math.isfinite(min_confidence) or not 0.0 <= min_confidence <= 1.0:
        raise ValueError("Minimum detection confidence must be in [0, 1].")

    frames = _frame_index(tracks, min_confidence)
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame_number, objects in frames.items():
        motorcycles = [(track_id, point) for track_id, (name, point) in objects.items() if name in MOTORCYCLE_CLASSES]
        riders = [(track_id, point) for track_id, (name, point) in objects.items() if name in RIDER_CLASSES]
        for motorcycle_id, motorcycle in motorcycles:
            bike_width = float(motorcycle["x2"]) - float(motorcycle["x1"])
            bike_height = float(motorcycle["y2"]) - float(motorcycle["y1"])
            if bike_width <= 0 or bike_height <= 0:
                continue
            associated = []
            for rider_id, rider in riders:
                rider_center_x = (float(rider["x1"]) + float(rider["x2"])) / 2
                motorcycle_center_x = (float(motorcycle["x1"]) + float(motorcycle["x2"])) / 2
                rider_center_y = (float(rider["y1"]) + float(rider["y2"])) / 2
                vertical_gap = max(
                    float(motorcycle["y1"]) - float(rider["y2"]),
                    float(rider["y1"]) - float(motorcycle["y2"]),
                    0.0,
                )
                if (
                    abs(rider_center_x - motorcycle_center_x) <= bike_width * 0.35
                    and vertical_gap <= bike_height * 0.12
                    and rider_center_y <= float(motorcycle["y2"]) + bike_height * 0.15
                ):
                    associated.append((rider_id, rider))
            associated = _distinct_riders(associated)
            if len({rider_id for rider_id, _ in associated}) >= min_riders:
                samples[motorcycle_id].append({
                    "frame": frame_number,
                    "motorcycle": motorcycle,
                    "riders": associated,
                })

    events = []
    for motorcycle_id, candidate_frames in samples.items():
        if len(candidate_frames) < min_candidate_frames:
            continue
        candidate_frames.sort(key=lambda item: (
            float(item["motorcycle"]["timestamp_seconds"]), item["frame"],
        ))
        duration = (
            float(candidate_frames[-1]["motorcycle"]["timestamp_seconds"])
            - float(candidate_frames[0]["motorcycle"]["timestamp_seconds"])
        )
        if duration < min_candidate_duration_seconds:
            continue
        evidence = max(
            candidate_frames,
            key=lambda item: (
                len(item["riders"]),
                sum(float(rider["confidence"]) for _, rider in item["riders"]),
                -item["frame"],
            ),
        )
        motorcycle = evidence["motorcycle"]
        riders = evidence["riders"]
        confidence_values = [float(motorcycle["confidence"])] + [
            float(rider["confidence"]) for _, rider in riders
        ]
        events.append({
            "event_id": f"triple_riding_candidate_track_{motorcycle_id}",
            "event_type": "triple_riding_candidate",
            "track_id": int(float(motorcycle_id)),
            "related_track_ids": sorted(int(float(rider_id)) for rider_id, _ in riders),
            "class_name": _class_name(tracks[motorcycle_id]),
            "timestamp_seconds": round(float(motorcycle["timestamp_seconds"]), 3),
            "frame_number": evidence["frame"],
            "evidence_frame": evidence["frame"],
            "start_seconds": round(float(candidate_frames[0]["motorcycle"]["timestamp_seconds"]), 3),
            "end_seconds": round(float(candidate_frames[-1]["motorcycle"]["timestamp_seconds"]), 3),
            "duration_seconds": round(max(0.0, duration), 3),
            "confidence": round(sum(confidence_values) / len(confidence_values), 4),
            "observations": len(candidate_frames),
            "participant_count": len({rider_id for rider_id, _ in riders}),
            "status": "needs_human_review",
            "description": f"At least {min_riders} rider-class tracks were repeatedly associated with one motorcycle track.",
            "notes": (
                "This image-space association can include nearby road users or tracking errors. "
                "Verify every rider and the motorcycle in the source video; this is not a confirmed violation."
            ),
            "evidence_image": "",
        })
    return sorted(events, key=lambda event: (event["start_seconds"], event["track_id"]))


def _distance_to_box(x: float, y: float, box: dict[str, Any]) -> float:
    dx = max(float(box["x1"]) - x, 0.0, x - float(box["x2"]))
    dy = max(float(box["y1"]) - y, 0.0, y - float(box["y2"]))
    return math.hypot(dx, dy)


def analyze_pedestrian_interaction_candidates(
    tracks: dict[str, list[dict[str, Any]]],
    *,
    min_shared_frames: int = 3,
    min_confidence: float = 0.35,
    min_person_displacement_heights: float = 0.4,
) -> list[dict[str, Any]]:
    """Flag moving pedestrians with sustained, closing vehicle proximity in image coordinates."""
    if min_shared_frames < 3:
        raise ValueError("Pedestrian interaction analysis requires at least three shared frames.")
    if not math.isfinite(min_confidence) or not 0.0 <= min_confidence <= 1.0:
        raise ValueError("Minimum detection confidence must be in [0, 1].")
    if (
        not math.isfinite(min_person_displacement_heights)
        or min_person_displacement_heights <= 0
    ):
        raise ValueError("Minimum pedestrian displacement must be finite and positive.")

    normalized = {
        str(track_id): [
            point for point in points
            if isinstance(point, dict) and _valid_point(point) and float(point["confidence"]) >= min_confidence
        ]
        for track_id, points in tracks.items()
    }
    people = {
        track_id: points for track_id, points in normalized.items()
        if points and _class_name(points) in PERSON_CLASSES
    }
    vehicles = {
        track_id: points for track_id, points in normalized.items()
        if points and _class_name(points) in VEHICLE_CLASSES
    }

    events = []
    for person_id, person_points in people.items():
        by_frame = {
            int(point.get("frame", point.get("frame_number", 0))): point
            for point in person_points
        }
        for vehicle_id, vehicle_points in vehicles.items():
            if person_id == vehicle_id:
                continue
            vehicle_by_frame = {
                int(point.get("frame", point.get("frame_number", 0))): point
                for point in vehicle_points
            }
            shared_frames = sorted(set(by_frame) & set(vehicle_by_frame))
            if len(shared_frames) < min_shared_frames:
                continue
            paired = []
            for frame in shared_frames:
                person = by_frame[frame]
                vehicle = vehicle_by_frame[frame]
                foot_x = (float(person["x1"]) + float(person["x2"])) / 2
                foot_y = float(person["y2"])
                person_height = float(person["y2"]) - float(person["y1"])
                distance = _distance_to_box(foot_x, foot_y, vehicle)
                paired.append((frame, person, vehicle, distance, person_height))
            paired.sort(key=lambda item: (float(item[1]["timestamp_seconds"]), item[0]))
            first_window = paired[:max(1, len(paired) // 3)]
            last_window = paired[-max(1, len(paired) // 3):]
            initial_distance = statistics.median(item[3] for item in first_window)
            final_distance = statistics.median(item[3] for item in last_window)
            typical_height = statistics.median(item[4] for item in paired)
            proximity_threshold = max(18.0, typical_height * 0.4)
            approach_threshold = max(10.0, typical_height * 0.15)
            first_foot_x = statistics.median(
                (float(item[1]["x1"]) + float(item[1]["x2"])) / 2
                for item in first_window
            )
            first_foot_y = statistics.median(float(item[1]["y2"]) for item in first_window)
            last_foot_x = statistics.median(
                (float(item[1]["x1"]) + float(item[1]["x2"])) / 2
                for item in last_window
            )
            last_foot_y = statistics.median(float(item[1]["y2"]) for item in last_window)
            pedestrian_displacement = math.hypot(
                last_foot_x - first_foot_x,
                last_foot_y - first_foot_y,
            )
            displacement_heights = (
                pedestrian_displacement / typical_height if typical_height > 0 else 0.0
            )
            if (
                final_distance > proximity_threshold
                or initial_distance - final_distance < approach_threshold
                or displacement_heights < min_person_displacement_heights
            ):
                continue

            evidence = min(paired, key=lambda item: item[3])
            _, person, vehicle, final_distance, _ = evidence
            time_start = float(paired[0][1]["timestamp_seconds"])
            time_end = float(paired[-1][1]["timestamp_seconds"])
            if time_end - time_start < 0.2:
                continue
            confidence_values = [float(item[1]["confidence"]) for item in paired]
            confidence_values.extend(float(item[2]["confidence"]) for item in paired)
            events.append({
                "event_id": f"pedestrian_interaction_candidate_{person_id}_{vehicle_id}",
                "event_type": "unsafe_pedestrian_interaction_candidate",
                "track_id": int(float(person_id)),
                "related_track_ids": [int(float(vehicle_id))],
                "class_name": _class_name(person_points),
                "vehicle_track_id": int(float(vehicle_id)),
                "vehicle_class_name": _class_name(vehicle_points),
                "timestamp_seconds": round(float(person["timestamp_seconds"]), 3),
                "frame_number": int(evidence[0]),
                "evidence_frame": int(evidence[0]),
                "start_seconds": round(time_start, 3),
                "end_seconds": round(time_end, 3),
                "duration_seconds": round(time_end - time_start, 3),
                "confidence": round(sum(confidence_values) / len(confidence_values), 4),
                "observations": len(paired),
                "image_space_separation_pixels": round(final_distance, 1),
                "person_displacement_pixels": round(pedestrian_displacement, 1),
                "person_displacement_heights": round(displacement_heights, 3),
                "status": "needs_human_review",
                "description": (
                    f"Pedestrian track {person_id} moved during a close interaction with "
                    f"{_class_name(vehicle_points)} track {vehicle_id}."
                ),
                "notes": (
                    "Pixel separation is not physical distance or time-to-collision. Perspective, "
                    "bounding boxes, and tracking errors can create false candidates; review the source video."
                ),
                "evidence_image": "",
            })
    return sorted(events, key=lambda event: (event["start_seconds"], event["track_id"], event["event_id"]))
