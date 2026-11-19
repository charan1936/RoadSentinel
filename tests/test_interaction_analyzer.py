import unittest

from tools.roadsentinel_interaction_analyzer import (
    analyze_pedestrian_interaction_candidates,
    analyze_triple_riding_candidates,
)


def point(frame, timestamp, class_name, box, confidence=0.9):
    x1, y1, x2, y2 = box
    return {
        "frame": frame,
        "timestamp_seconds": timestamp,
        "class_name": class_name,
        "confidence": confidence,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


class InteractionAnalyzerTests(unittest.TestCase):
    def test_triple_riding_requires_repeated_rider_associations(self):
        tracks = {
            "1": [point(frame, frame * 0.2, "motorcycle", (100, 100, 200, 180)) for frame in range(6)],
            "2": [point(frame, frame * 0.2, "rider", (112, 55, 135, 115)) for frame in range(6)],
            "3": [point(frame, frame * 0.2, "rider", (140, 50, 165, 115)) for frame in range(6)],
            "4": [point(frame, frame * 0.2, "rider", (168, 60, 190, 118)) for frame in range(6)],
        }

        events = analyze_triple_riding_candidates(tracks)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "triple_riding_candidate")
        self.assertEqual(events[0]["participant_count"], 3)
        self.assertEqual(events[0]["related_track_ids"], [2, 3, 4])
        self.assertEqual(events[0]["status"], "needs_human_review")

    def test_triple_riding_rejects_nonpersistent_or_unassociated_people(self):
        tracks = {
            "1": [point(frame, frame * 0.2, "motorcycle", (100, 100, 200, 180)) for frame in range(6)],
            "2": [point(frame, frame * 0.2, "rider", (400, 55, 430, 115)) for frame in range(6)],
            "3": [point(frame, frame * 0.2, "rider", (440, 50, 470, 115)) for frame in range(6)],
            "4": [point(frame, frame * 0.2, "rider", (480, 60, 510, 118)) for frame in range(6)],
        }

        self.assertEqual(analyze_triple_riding_candidates(tracks), [])

    def test_triple_riding_ignores_duplicate_and_nearby_rider_boxes(self):
        tracks = {
            "1": [point(frame, frame * 0.2, "motorcycle", (100, 100, 200, 180)) for frame in range(6)],
            "2": [point(frame, frame * 0.2, "rider", (112, 55, 135, 115)) for frame in range(6)],
            "3": [point(frame, frame * 0.2, "rider", (113, 56, 136, 116)) for frame in range(6)],
            "4": [point(frame, frame * 0.2, "rider", (140, 50, 165, 115)) for frame in range(6)],
            "5": [point(frame, frame * 0.2, "rider", (260, 55, 285, 115)) for frame in range(6)],
            "6": [point(frame, frame * 0.2, "rider", (140, 195, 165, 245)) for frame in range(6)],
        }

        self.assertEqual(analyze_triple_riding_candidates(tracks), [])

    def test_triple_riding_requires_sustained_candidates(self):
        tracks = {
            "1": [point(frame, frame * 0.1, "motorcycle", (100, 100, 200, 180)) for frame in range(6)],
            "2": [point(frame, frame * 0.1, "rider", (112, 55, 135, 115)) for frame in range(6)],
            "3": [point(frame, frame * 0.1, "rider", (140, 50, 165, 115)) for frame in range(6)],
            "4": [point(frame, frame * 0.1, "rider", (168, 60, 190, 118)) for frame in range(6)],
        }

        self.assertEqual(analyze_triple_riding_candidates(tracks), [])

    def test_triple_riding_participant_count_matches_evidence_frame(self):
        tracks = {
            "1": [point(frame, frame * 0.2, "motorcycle", (100, 100, 200, 180)) for frame in range(6)],
            "2": [point(frame, frame * 0.2, "rider", (112, 55, 135, 115)) for frame in range(6)],
            "3": [point(frame, frame * 0.2, "rider", (140, 50, 165, 115)) for frame in range(6)],
            "4": [point(frame, frame * 0.2, "rider", (168, 60, 190, 118)) for frame in range(6)],
            "5": [point(frame, frame * 0.2, "rider", (120, 45, 145, 110)) for frame in range(2)],
        }

        event = analyze_triple_riding_candidates(tracks)[0]

        self.assertEqual(event["evidence_frame"], 0)
        self.assertEqual(event["participant_count"], len(event["related_track_ids"]))
        self.assertEqual(event["participant_count"], 4)

    def test_pedestrian_candidate_requires_close_approach(self):
        tracks = {
            "10": [point(frame, frame * 0.3, "person", (x, 70, x + 24, 130))
                   for frame, x in enumerate((250, 220, 190, 160))],
            "20": [point(frame, frame * 0.3, "car", (100, 100, 150, 150))
                   for frame in range(4)],
        }

        events = analyze_pedestrian_interaction_candidates(tracks)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "unsafe_pedestrian_interaction_candidate")
        self.assertEqual(events[0]["track_id"], 10)
        self.assertEqual(events[0]["class_name"], "person")
        self.assertEqual(events[0]["vehicle_track_id"], 20)
        self.assertEqual(events[0]["vehicle_class_name"], "car")
        self.assertEqual(events[0]["related_track_ids"], [20])
        self.assertGreaterEqual(events[0]["person_displacement_heights"], 0.4)
        self.assertIn("Pixel separation is not physical distance", events[0]["notes"])

    def test_pedestrian_candidate_rejects_static_proximity(self):
        tracks = {
            "10": [point(frame, frame * 0.3, "person", (160, 70, 184, 130))
                   for frame in range(4)],
            "20": [point(frame, frame * 0.3, "car", (100, 100, 150, 150))
                   for frame in range(4)],
        }

        self.assertEqual(analyze_pedestrian_interaction_candidates(tracks), [])

    def test_moving_vehicle_does_not_create_candidate_for_stationary_pedestrian(self):
        tracks = {
            "10": [point(frame, frame * 0.3, "person", (150, 70, 174, 130))
                   for frame in range(4)],
            "20": [
                point(frame, frame * 0.3, "truck", box)
                for frame, box in enumerate((
                    (240, 100, 280, 150),
                    (220, 100, 260, 150),
                    (190, 100, 230, 150),
                    (170, 100, 210, 150),
                ))
            ],
        }

        self.assertEqual(analyze_pedestrian_interaction_candidates(tracks), [])


if __name__ == "__main__":
    unittest.main()
