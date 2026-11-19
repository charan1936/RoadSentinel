import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.roadsentinel_directional_event_analyzer import analyze_wrong_way_tracks


class DirectionalEvidenceTests(unittest.TestCase):
    def test_wrong_way_event_has_bbox_and_mid_track_evidence_frame(self):
        points = []
        for i in range(10):
            cy = 800 - i * 33
            points.append({
                "frame": i * 10, "timestamp_seconds": i * 0.4,
                "class_name": "motorcycle", "confidence": 0.8,
                "x1": 480, "y1": cy - 20, "x2": 520, "y2": cy + 20,
                "cx": 500, "cy": cy, "width": 40, "height": 40,
            })
        calibration = {
            "roi_polygon": [{"x": .1, "y": .1}, {"x": .9, "y": .1}, {"x": .9, "y": .9}, {"x": .1, "y": .9}],
            "direction_start": {"x": .5, "y": .2},
            "direction_end": {"x": .5, "y": .8},
        }
        events = analyze_wrong_way_tracks({"7": points}, calibration, 1000, 1000)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(len(event["evidence_bbox"]), 4)
        self.assertGreater(event["evidence_frame"], event["frame_number"])
        self.assertLessEqual(event["evidence_frame"], 90)


if __name__ == "__main__":
    unittest.main()
    