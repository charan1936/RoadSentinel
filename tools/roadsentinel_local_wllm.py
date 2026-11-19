"""
RoadSentinel Local Grounded WLLM Engine
Generates schema-compliant audit summaries, event explanations, and Q&A responses
directly from structured video telemetry without external API dependencies.
"""

def local_ai_summary(source: dict, events: list, included: list, event_data: list, type_counts: dict, status_counts: dict) -> dict:
    uploaded_name = source.get("uploaded_filename", "uploaded video")
    total_events = len(events)
    wrong_way_cnt = type_counts.get("wrong_way_movement_candidate", 0)
    parking_cnt = type_counts.get("stationary_vehicle_candidate", 0) + type_counts.get("restricted_parking_candidate", 0)
    movement_cnt = type_counts.get("movement_observation", 0)
    needs_review = [e for e in event_data if e.get("status") in ("needs_human_review", "candidate_violation")]
    reviewed = [e for e in event_data if e.get("status") in ("confirmed_by_reviewer", "rejected_by_reviewer")]

    summary_text = (
        f"RoadSentinel telemetry audit for {uploaded_name}: Analyzed {total_events} total tracked events. "
        f"Identified {len(needs_review)} candidate events requiring human review, {len(reviewed)} reviewed decisions, "
        f"and {movement_cnt} baseline traffic movement tracks. "
        f"Screening identified {wrong_way_cnt} directional anomalies and {parking_cnt} stationary/restricted-zone candidates."
    )
    patterns = []
    if wrong_way_cnt > 0:
        patterns.append(f"{wrong_way_cnt} tracks exhibited displacement vectors opposing the calibrated road direction.")
    if parking_cnt > 0:
        patterns.append(f"{parking_cnt} vehicles exhibited stationary kinematic profiles inside restricted ROI boundaries.")
    if movement_cnt > 0:
        patterns.append(f"{movement_cnt} vehicle tracks followed standard movement profiles without geometric violations.")
    if not patterns:
        patterns.append("No anomalous vehicle trajectories detected under current candidate rules.")

    events_to_review = []
    for e in needs_review:
        eid = str(e.get("event_id"))
        cls = str(e.get("class_name", "Vehicle")).capitalize()
        etype = str(e.get("event_type", "candidate")).replace("_", " ")
        t_sec = float(e.get("timestamp_seconds") or 0)
        events_to_review.append({
            "event_id": eid,
            "reason": f"{cls} (track {e.get('track_id')}) triggered {etype} at {t_sec:.1f}s."
        })

    reviewed_events = []
    for e in reviewed:
        eid = str(e.get("event_id"))
        status = e.get("status")
        note = str(e.get("review_note") or "Decision logged in audit trail.")
        reviewed_events.append({
            "event_id": eid,
            "review_status": status,
            "note": note
        })

    limitations = [
        "Dataset (IDD) fine-tuning provides road object localization; it does not detect helmetless riders, triple riding, or red-light violations.",
        "Monocular perspective limits metric speed estimation; pixel displacement rates are uncalibrated spatial heuristics.",
        "Stationary candidates cannot distinguish temporary traffic congestion from illegal curb parking without human visual confirmation."
    ]

    return {
        "executive_summary": summary_text,
        "observed_patterns": patterns,
        "events_to_review": events_to_review,
        "reviewed_events": reviewed_events,
        "limitations": limitations,
    }


def local_event_explanation(event_data: dict) -> dict:
    cls_name = str(event_data.get("class_name", "vehicle")).capitalize()
    etype = str(event_data.get("event_type", "candidate event")).replace("_", " ")
    status = str(event_data.get("status", "needs_human_review"))
    track_id = event_data.get("track_id", "N/A")
    t_sec = float(event_data.get("timestamp_seconds") or 0)
    dur = float(event_data.get("duration_seconds") or 0)
    conf = float(event_data.get("confidence") or 0)

    explanation = (
        f"{cls_name} (track {track_id}) was flagged as a {etype} at timestamp {t_sec:.2f}s "
        f"with an observation duration of {dur:.2f} seconds. Current status is '{status}'."
    )
    facts = [
        f"Class: {cls_name}",
        f"Track ID: {track_id}",
        f"Timestamp: {t_sec:.2f}s (Frame {event_data.get('frame_number', 'N/A')})",
        f"Detection confidence: {conf:.2f}",
        f"Duration: {dur:.2f}s across {event_data.get('observations', 1)} observed frames"
    ]
    uncertainties = [
        "Monocular bounding box estimation is susceptible to visual perspective distortion and temporary occlusions.",
        "Tracking ID continuity may be affected by vehicles passing behind obstacles or trees.",
        "Automated candidate event generation is an audit alert, not an authoritative legal citation."
    ]
    guidance = (
        f"Review original footage around timestamp {t_sec:.1f}s (frame {event_data.get('frame_number', 'N/A')}) "
        f"to confirm whether vehicle behavior constitutes an infraction or acceptable traffic flow before updating status."
    )
    return {
        "explanation": explanation,
        "supporting_facts": facts,
        "uncertainties": uncertainties,
        "reviewer_guidance": guidance
    }


def local_qa(question: str, event_data: list, events: list) -> dict:
    q = question.lower()
    unsupported_keywords = ["helmet", "triple", "red light", "signal", "traffic light", "phone", "overload", "seatbelt"]
    if any(k in q for k in unsupported_keywords):
        return {
            "answer": "RoadSentinel's vision pipeline utilizes the Indian Driving Dataset (IDD) for vehicle localization and geometric ROI heuristics for wrong-way and stationary tracking. The current model and telemetry do not record helmet usage, passenger counts, or traffic signal states, so data is insufficient to evaluate this query.",
            "citations": [],
            "insufficient_data": True
        }

    if any(k in q for k in ["wrong", "reverse", "opposite", "direction", "counter"]):
        matches = [e for e in event_data if "wrong_way" in str(e.get("event_type", ""))]
        citations = [
            {"event_id": str(e["event_id"]), "reason": f"{str(e.get('class_name')).capitalize()} track {e.get('track_id')} traveled against calibrated road vector at {float(e.get('timestamp_seconds') or 0):.1f}s."}
            for e in matches[:8]
        ]
        return {
            "answer": f"Found {len(matches)} wrong-way candidate event(s) in this video. These tracks exhibited displacement vectors opposing the calibrated road direction.",
            "citations": citations,
            "insufficient_data": False
        }

    if any(k in q for k in ["park", "stationary", "stopped", "halt", "curb", "standing"]):
        matches = [e for e in event_data if any(k in str(e.get("event_type", "")) for k in ["parking", "stationary"])]
        citations = [
            {"event_id": str(e["event_id"]), "reason": f"{str(e.get('class_name')).capitalize()} track {e.get('track_id')} remained stationary inside restricted zone at {float(e.get('timestamp_seconds') or 0):.1f}s."}
            for e in matches[:8]
        ]
        return {
            "answer": f"Found {len(matches)} stationary or restricted-zone parking candidate(s) in this video job.",
            "citations": citations,
            "insufficient_data": False
        }

    total = len(events)
    candidates = [e for e in event_data if e.get("status") in ("needs_human_review", "candidate_violation")]
    citations = [
        {"event_id": str(e["event_id"]), "reason": f"{str(e.get('class_name')).capitalize()} track {e.get('track_id')} ({e.get('event_type')}) at {float(e.get('timestamp_seconds') or 0):.1f}s."}
        for e in candidates[:5]
    ]
    return {
        "answer": f"Video contains {total} total telemetry events across {len(event_data)} analyzed tracks, including {len(candidates)} candidate event(s) requiring human operator review.",
        "citations": citations,
        "insufficient_data": False
    }
