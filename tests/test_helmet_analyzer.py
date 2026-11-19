import unittest

from tools.roadsentinel_helmet_analyzer import analyze_helmet_candidates, validate_helmet_model_classes


MODEL_CLASSES = {0: "with helmet", 1: "without helmet", 2: "rider", 3: "number plate"}


def detection(frame, timestamp, class_name, box, track_id, confidence=0.9):
    x1, y1, x2, y2 = box
    return {
        "frame": frame,
        "timestamp_seconds": timestamp,
        "class_name": class_name,
        "confidence": confidence,
        "track_id": track_id,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


class HelmetAnalyzerTests(unittest.TestCase):
    def test_requires_explicit_helmet_and_no_helmet_model_classes(self):
        with self.assertRaisesRegex(ValueError, "explicit rider"):
            validate_helmet_model_classes({0: "helmet", 1: "person"})

    def test_repeated_explicit_no_helmet_detections_generate_review_candidate(self):
        detections = []
        primary_riders = []
        for frame, timestamp in enumerate((0.0, 0.3, 0.6)):
            detections.extend([
                detection(frame, timestamp, "rider", (100, 100, 200, 300), 11),
                detection(frame, timestamp, "without helmet", (140, 105, 160, 130), 30 + frame, 0.8),
            ])
            primary_riders.append(detection(frame, timestamp, "rider", (100, 100, 200, 300), 91))

        events, diagnostics = analyze_helmet_candidates(
            detections,
            MODEL_CLASSES,
            primary_rider_detections=primary_riders,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "helmetless_riding_candidate")
        self.assertEqual(events[0]["track_id"], 91)
        self.assertEqual(events[0]["helmet_model_track_id"], 11)
        self.assertEqual(events[0]["track_source"], "primary_rider_cross_checked")
        self.assertEqual(events[0]["observations"], 3)
        self.assertEqual(events[0]["timestamp_seconds"], 0.0)
        self.assertEqual(events[0]["status"], "needs_human_review")
        self.assertEqual(events[0]["confidence"], 0.8)
        self.assertEqual(diagnostics["unmatched_status_detections"], 0)

    def test_absence_of_helmet_detection_is_not_a_helmetless_candidate(self):
        detections = [
            detection(frame, frame * 0.3, "rider", (100, 100, 200, 300), 11)
            for frame in range(5)
        ]

        events, _ = analyze_helmet_candidates(detections, MODEL_CLASSES)

        self.assertEqual(events, [])

    def test_specialist_rider_on_primary_person_is_not_a_helmet_candidate(self):
        detections = []
        primary_people = []
        for frame, timestamp in enumerate((0.0, 0.3, 0.6)):
            detections.extend([
                detection(frame, timestamp, "rider", (100, 100, 200, 300), 11),
                detection(frame, timestamp, "without helmet", (140, 105, 160, 130), 30 + frame, 0.8),
            ])
            primary_people.append(detection(frame, timestamp, "person", (100, 100, 200, 300), 91))

        events, diagnostics = analyze_helmet_candidates(
            detections,
            MODEL_CLASSES,
            primary_rider_detections=primary_people,
        )

        self.assertEqual(events, [])
        self.assertEqual(diagnostics["uncorroborated_rider_detections"], 3)

    def test_specialist_rider_requires_primary_detector_corroboration(self):
        detections = []
        for frame, timestamp in enumerate((0.0, 0.3, 0.6)):
            detections.extend([
                detection(frame, timestamp, "rider", (100, 100, 200, 300), 11),
                detection(frame, timestamp, "without helmet", (140, 105, 160, 130), 30 + frame, 0.8),
            ])

        events, diagnostics = analyze_helmet_candidates(detections, MODEL_CLASSES)

        self.assertEqual(events, [])
        self.assertEqual(diagnostics["uncorroborated_rider_detections"], 3)

    def test_unmatched_no_helmet_detections_are_not_assigned_to_nearby_people(self):
        detections = []
        for frame, timestamp in enumerate((0.0, 0.3, 0.6)):
            detections.extend([
                detection(frame, timestamp, "rider", (300, 100, 400, 300), 11),
                detection(frame, timestamp, "without helmet", (100, 105, 120, 130), 30 + frame),
            ])

        events, diagnostics = analyze_helmet_candidates(detections, MODEL_CLASSES)

        self.assertEqual(events, [])
        self.assertEqual(diagnostics["unmatched_status_detections"], 3)

    def test_conflicting_helmet_observations_do_not_create_candidate(self):
        detections = []
        for frame, timestamp in enumerate((0.0, 0.3, 0.6, 0.9)):
            detections.append(detection(frame, timestamp, "rider", (100, 100, 200, 300), 11))
            label = "without helmet" if frame < 2 else "with helmet"
            detections.append(detection(frame, timestamp, label, (140, 105, 160, 130), 30 + frame, 0.8))

        events, _ = analyze_helmet_candidates(detections, MODEL_CLASSES)

        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
