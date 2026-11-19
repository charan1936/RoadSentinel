import unittest

from tools.roadsentinel_parking_event_analyzer import analyze_parking_zone_tracks


def points(class_name="car", centers=None, timestamps=None, track_id=7):
    centers = centers or [(100.0, 100.0)] * 11
    timestamps = timestamps or [float(index) for index in range(len(centers))]
    return {
        str(track_id): [
            {
                "frame": index,
                "timestamp_seconds": timestamp,
                "class_name": class_name,
                "confidence": 0.9,
                "x1": center_x - 10,
                "y1": center_y - 10,
                "x2": center_x + 10,
                "y2": center_y + 10,
            }
            for index, (center, timestamp) in enumerate(zip(centers, timestamps))
            for center_x, center_y in [center]
        ],
    }


ZONE = [
    {"x": 0.1, "y": 0.1},
    {"x": 0.8, "y": 0.1},
    {"x": 0.8, "y": 0.8},
    {"x": 0.1, "y": 0.8},
]


class ParkingAnalyzerTests(unittest.TestCase):
    def test_stationary_vehicle_in_zone_generates_review_candidate(self):
        track_points = [(100.0 + (index % 2), 100.0) for index in range(11)]

        events = analyze_parking_zone_tracks(
            points(centers=track_points),
            ZONE,
            width=200,
            height=200,
            min_stationary_seconds=10.0,
        )

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["event_type"], "parking_zone_candidate")
        self.assertEqual(event["status"], "needs_human_review")
        self.assertEqual(event["track_id"], "7")
        self.assertEqual(event["class_name"], "car")
        self.assertEqual(event["start_seconds"], 0.0)
        self.assertEqual(event["end_seconds"], 10.0)
        self.assertTrue(event["evidence_bbox"])
        self.assertIn("not proof of illegal parking", " ".join(event["notes"]))

    def test_moving_vehicle_does_not_generate_parking_candidate(self):
        moving_centers = [(40.0 + index * 10, 100.0) for index in range(11)]

        events = analyze_parking_zone_tracks(
            points(centers=moving_centers),
            ZONE,
            width=200,
            height=200,
            min_stationary_seconds=10.0,
        )

        self.assertEqual(events, [])

    def test_vehicle_outside_zone_does_not_generate_candidate(self):
        outside_centers = [(190.0, 190.0)] * 11

        events = analyze_parking_zone_tracks(
            points(centers=outside_centers),
            ZONE,
            width=200,
            height=200,
            min_stationary_seconds=10.0,
        )

        self.assertEqual(events, [])

    def test_invalid_frame_size_and_polygon_are_rejected(self):
        with self.assertRaises(ValueError):
            analyze_parking_zone_tracks({}, ZONE, width=0, height=200)
        with self.assertRaises(ValueError):
            analyze_parking_zone_tracks({}, [{"x": 0.1, "y": 0.1}], width=200, height=200)


if __name__ == "__main__":
    unittest.main()
