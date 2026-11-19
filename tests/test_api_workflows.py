import json
import asyncio
import csv
import io
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from starlette.datastructures import UploadFile

import api
from tools.roadsentinel_candidate_event_analyzer import analyze_track, load_tracks
from tools.roadsentinel_directional_event_analyzer import analyze_wrong_way_tracks


def provider_response(payload):
    return {
        "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": json.dumps(payload)}],
        }]
    }


class ApiWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.scratch.name)
        self.patches = ExitStack()
        self.patches.enter_context(patch.object(api, "VIDEO_JOBS_DB_PATH", self.root / "jobs.sqlite3"))
        self.patches.enter_context(patch.object(api, "AI_QA_DB_PATH", self.root / "qa.sqlite3"))
        self.patches.enter_context(patch.object(api, "REVIEW_DB_PATH", self.root / "reviews.sqlite3"))
        self.patches.enter_context(patch.object(api, "JOBS_DIR", self.root / "video_jobs"))
        self.patches.enter_context(patch.object(api, "JOB_EVIDENCE_DIR", self.root / "video_evidence"))
        self.patches.enter_context(patch.object(api, "UPLOADS_DIR", self.root / "uploads"))
        self.patches.enter_context(patch.object(api, "jobs", {}))
        self.patches.enter_context(patch.object(api, "job_persistence_state", {}))
        self.patches.enter_context(patch.object(api, "model_cache", {}))
        self.patches.enter_context(patch.object(
            api,
            "get_backend_setting",
            side_effect=lambda name: {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_MODEL": "test-model",
            }.get(name, ""),
        ))

    def tearDown(self):
        self.patches.close()
        self.scratch.cleanup()

    def mock_provider(self, payload):
        client = self.patches.enter_context(patch.object(api.httpx, "Client")).return_value.__enter__.return_value
        client.post.return_value.raise_for_status.return_value = None
        client.post.return_value.json.return_value = provider_response(payload)
        return client

    def test_video_jobs_survive_restart_and_in_progress_job_is_marked_failed(self):
        api.save_persisted_video_job({
            "job_id": "123456abcdef", "status": "completed", "progress": 100,
            "stage": "Analysis complete", "original_filename": "finished.mp4",
            "created_at": "2026-10-30T10:00:00+00:00", "updated_at": "2026-10-30T10:01:00+00:00",
        })
        api.save_persisted_video_job({
            "job_id": "abcdef123456", "status": "processing", "progress": 41,
            "stage": "Analyzing frame 410", "original_filename": "interrupted.mp4",
            "created_at": "2026-10-30T10:00:00+00:00", "updated_at": "2026-10-30T10:01:00+00:00",
        })

        api.restore_persisted_video_jobs()

        self.assertEqual(api.jobs["123456abcdef"]["status"], "completed")
        self.assertEqual(api.jobs["abcdef123456"]["status"], "failed")
        self.assertIn("restart", api.jobs["abcdef123456"]["stage"].lower())

    def test_model_path_uses_default_or_backend_override(self):
        with patch.object(api, "get_backend_setting", return_value=""):
            self.assertEqual(api.get_model_path(), api.DEFAULT_MODEL_PATH)
        configured = self.root / "model weights" / "best.pt"
        with patch.object(api, "get_backend_setting", return_value=str(configured)):
            self.assertEqual(api.get_model_path(), configured)

    def test_tracking_settings_are_configurable_and_validated(self):
        with patch.object(api, "get_backend_setting", return_value=""):
            self.assertEqual(api.get_tracking_settings(), {"confidence": 0.45, "imgsz": 960, "tracker_config": "botsort.yaml"})
        settings = {"ROAD_SENTINEL_TRACK_CONFIDENCE": "0.25", "ROAD_SENTINEL_TRACKER_CONFIG": "tools/experiments/botsort_roadsentinel.yaml"}
        with patch.object(api, "get_backend_setting", side_effect=lambda name: settings.get(name, "")):
            expected_tracker = str((api.ROOT / settings["ROAD_SENTINEL_TRACKER_CONFIG"]).resolve())
            self.assertEqual(api.get_tracking_settings(), {"confidence": 0.25, "imgsz": 960, "tracker_config": expected_tracker})
        with patch.object(api, "get_backend_setting", side_effect=lambda name: "1.1" if name == "ROAD_SENTINEL_TRACK_CONFIDENCE" else ""):
            with self.assertRaises(ValueError):
                api.get_tracking_settings()
        with patch.object(api, "get_backend_setting", side_effect=lambda name: "missing-tracker.yaml" if name == "ROAD_SENTINEL_TRACKER_CONFIG" else ""):
            with self.assertRaises(ValueError):
                api.get_tracking_settings()

    def test_event_settings_are_configurable_and_validated(self):
        defaults = {
            "heuristic_version": "1.0",
            "min_duration_seconds": 3.0,
            "stationary_speed_px_per_second": 5.0,
            "min_detection_confidence": 0.35,
        }
        with patch.object(api, "get_backend_setting", return_value=""):
            self.assertEqual(api.get_event_settings(), defaults)
        configured = {
            "ROAD_SENTINEL_EVENT_MIN_DURATION_SECONDS": "4.5",
            "ROAD_SENTINEL_EVENT_STATIONARY_SPEED_PX_S": "7",
            "ROAD_SENTINEL_EVENT_MIN_CONFIDENCE": "0.4",
        }
        with patch.object(api, "get_backend_setting", side_effect=lambda name: configured.get(name, "")):
            settings = api.get_event_settings()
        self.assertEqual(settings["min_duration_seconds"], 4.5)
        self.assertEqual(settings["stationary_speed_px_per_second"], 7.0)
        self.assertEqual(settings["min_detection_confidence"], 0.4)
        with patch.object(api, "get_backend_setting", side_effect=lambda name: "1.2" if name == "ROAD_SENTINEL_EVENT_MIN_CONFIDENCE" else ""):
            with self.assertRaises(ValueError):
                api.get_event_settings()

    def test_helmet_settings_are_configurable_and_validated(self):
        defaults = {
            "ROAD_SENTINEL_HELMET_MODEL_PATH": "D:\\helmet.pt",
            "ROAD_SENTINEL_HELMET_CONFIDENCE": "0.7",
            "ROAD_SENTINEL_HELMET_FRAME_STRIDE": "5",
            "ROAD_SENTINEL_HELMET_MIN_OBSERVATIONS": "4",
            "ROAD_SENTINEL_HELMET_MIN_DURATION_SECONDS": "0.8",
            "ROAD_SENTINEL_HELMET_MIN_NO_HELMET_RATIO": "0.75",
        }
        with patch.object(api, "get_backend_setting", side_effect=lambda name: defaults.get(name, "")):
            settings = api.get_helmet_settings()
        self.assertEqual(settings["model_path"], Path("D:\\helmet.pt"))
        self.assertEqual(settings["confidence"], 0.7)
        self.assertEqual(settings["frame_stride"], 5)
        self.assertEqual(settings["min_observations"], 4)
        self.assertEqual(settings["min_duration_seconds"], 0.8)
        self.assertEqual(settings["min_no_helmet_ratio"], 0.75)
        with patch.object(api, "get_backend_setting", side_effect=lambda name: "31" if name == "ROAD_SENTINEL_HELMET_FRAME_STRIDE" else ""):
            with self.assertRaisesRegex(ValueError, "ROAD_SENTINEL_HELMET_FRAME_STRIDE"):
                api.get_helmet_settings()

    def test_event_engine_separates_stationary_from_moving_tracks(self):
        def point(frame, timestamp, center_x, confidence=0.8):
            return {
                "frame": frame, "timestamp_seconds": timestamp,
                "class_name": "car", "confidence": confidence,
                "x1": center_x - 5, "x2": center_x + 5,
                "y1": 95, "y2": 105,
                "cx": center_x, "cy": 100,
            }

        stationary = [point(i, float(i), x) for i, x in enumerate((99, 101, 100, 102, 99))]
        moving = [point(i, float(i), 40 + i * 15) for i in range(5)]
        low_confidence = [point(i, float(i), 100, 0.2) for i in range(5)]
        stationary_event = analyze_track("car-1", stationary, 3.0, 5.0, 0.35)[0]
        moving_event = analyze_track("car-2", moving, 3.0, 5.0, 0.35)[0]
        self.assertEqual(stationary_event["event_type"], "stationary_object_candidate")
        self.assertEqual(stationary_event["status"], "needs_human_review")
        self.assertEqual(moving_event["event_type"], "movement_observation")
        self.assertEqual(moving_event["status"], "observation")
        self.assertEqual(analyze_track("car-3", low_confidence, 3.0, 5.0, 0.35), [])

    def test_video_candidate_generation_integrates_interaction_events_and_evidence(self):
        api.JOBS_DIR.mkdir(parents=True, exist_ok=True)
        api.JOB_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        detections = self.root / "detections.csv"
        fields = ["frame", "timestamp_seconds", "track_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
        rows = []
        for frame, timestamp in enumerate((0.0, 0.3, 0.6, 0.9, 1.2, 1.5)):
            rows.extend([
                {"frame": frame, "timestamp_seconds": timestamp, "track_id": 1, "class_name": "motorcycle", "confidence": 0.9, "x1": 100, "y1": 100, "x2": 200, "y2": 180},
                {"frame": frame, "timestamp_seconds": timestamp, "track_id": 2, "class_name": "rider", "confidence": 0.9, "x1": 112, "y1": 55, "x2": 135, "y2": 115},
                {"frame": frame, "timestamp_seconds": timestamp, "track_id": 3, "class_name": "rider", "confidence": 0.9, "x1": 140, "y1": 50, "x2": 165, "y2": 115},
                {"frame": frame, "timestamp_seconds": timestamp, "track_id": 4, "class_name": "rider", "confidence": 0.9, "x1": 168, "y1": 60, "x2": 190, "y2": 118},
                {"frame": frame, "timestamp_seconds": timestamp, "track_id": 5, "class_name": "car", "confidence": 0.9, "x1": 100, "y1": 100, "x2": 150, "y2": 150},
                {"frame": frame, "timestamp_seconds": timestamp, "track_id": 6, "class_name": "person", "confidence": 0.9, "x1": 250 - frame * 30, "y1": 70, "x2": 274 - frame * 30, "y2": 130},
            ])
        with detections.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        class FakeCapture:
            def isOpened(self):
                return True

            def get(self, property_id):
                return 6 if property_id == api.cv2.CAP_PROP_FRAME_COUNT else 0

            def set(self, *_args):
                return True

            def read(self):
                return True, __import__("numpy").zeros((240, 320, 3), dtype="uint8")

            def release(self):
                pass

        with patch.object(api.cv2, "VideoCapture", return_value=FakeCapture()), patch.object(api.cv2, "imwrite", return_value=True):
            events, evidence_count = api.create_job_candidate_events(
                "123456abcdef", self.root / "traffic.mp4", detections,
            )

        event_types = {event["event_type"] for event in events}
        self.assertIn("triple_riding_candidate", event_types)
        self.assertIn("unsafe_pedestrian_interaction_candidate", event_types)
        pedestrian = next(
            event for event in events
            if event["event_type"] == "unsafe_pedestrian_interaction_candidate"
        )
        self.assertEqual(pedestrian["track_id"], 6)
        self.assertEqual(pedestrian["class_name"], "person")
        self.assertIn(
            (pedestrian["vehicle_track_id"], pedestrian["vehicle_class_name"]),
            {(1, "motorcycle"), (5, "car")},
        )
        self.assertEqual(evidence_count, 3)
        payload = json.loads((api.JOBS_DIR / "123456abcdef_candidate_events.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["category_analyses"]["triple_riding"]["status"], "events_detected")
        self.assertEqual(payload["category_analyses"]["unsafe_pedestrian_crossing"]["status"], "events_detected")

    def test_video_candidate_generation_integrates_helmet_candidates_and_evidence(self):
        api.JOBS_DIR.mkdir(parents=True, exist_ok=True)
        api.JOB_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        detections = self.root / "helmet_detections.csv"
        fields = ["frame", "timestamp_seconds", "track_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
        rows = [
            {"frame": frame, "timestamp_seconds": timestamp, "track_id": 1, "class_name": "rider",
             "confidence": 0.9, "x1": 100, "y1": 100, "x2": 200, "y2": 300}
            for frame, timestamp in enumerate((0.0, 0.3, 0.6))
        ]
        with detections.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        class FakeCapture:
            def isOpened(self):
                return True

            def get(self, property_id):
                return 3 if property_id == api.cv2.CAP_PROP_FRAME_COUNT else 0

            def set(self, *_args):
                return True

            def read(self):
                return True, __import__("numpy").zeros((360, 480, 3), dtype="uint8")

            def release(self):
                pass

        helmet_detections = []
        for frame, timestamp in enumerate((0.0, 0.3, 0.6)):
            helmet_detections.extend([
                {"frame": frame, "timestamp_seconds": timestamp, "track_id": 10,
                 "class_name": "rider", "confidence": 0.9,
                 "x1": 100, "y1": 100, "x2": 200, "y2": 300},
                {"frame": frame, "timestamp_seconds": timestamp, "track_id": 20 + frame,
                 "class_name": "without helmet", "confidence": 0.85,
                 "x1": 140, "y1": 105, "x2": 160, "y2": 130},
            ])

        with patch.object(api.cv2, "VideoCapture", return_value=FakeCapture()), patch.object(api.cv2, "imwrite", return_value=True):
            events, evidence_count = api.create_job_candidate_events(
                "123456abcdef",
                self.root / "traffic.mp4",
                detections,
                helmet_detections=helmet_detections,
                helmet_class_names={0: "with helmet", 1: "without helmet", 2: "rider"},
                helmet_settings={
                    "confidence": 0.65,
                    "min_observations": 3,
                    "min_duration_seconds": 0.5,
                    "min_no_helmet_ratio": 0.6,
                },
            )

        helmet_events = [event for event in events if event["event_type"] == "helmetless_riding_candidate"]
        self.assertEqual(len(helmet_events), 1)
        self.assertEqual(helmet_events[0]["track_id"], 1)
        self.assertEqual(helmet_events[0]["helmet_model_track_id"], 10)
        self.assertEqual(helmet_events[0]["track_source"], "primary_rider_cross_checked")
        self.assertEqual(helmet_events[0]["evidence_bbox"], [140.0, 105.0, 160.0, 130.0])
        self.assertEqual(evidence_count, 1)
        payload = json.loads((api.JOBS_DIR / "123456abcdef_candidate_events.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["category_analyses"]["helmetless_riding"]["status"], "events_detected")

    def test_empty_detection_file_produces_explicit_category_outcomes(self):
        job_id = "123456abcdef"
        api.JOBS_DIR.mkdir(parents=True, exist_ok=True)
        api.JOB_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        detections = self.root / "empty_detections.csv"
        detections.write_text(
            "frame,timestamp_seconds,track_id,class_id,class_name,confidence,x1,y1,x2,y2\n",
            encoding="utf-8",
        )

        class FakeCapture:
            def isOpened(self):
                return True

            def get(self, _property_id):
                return 0

            def release(self):
                pass

        with patch.object(api.cv2, "VideoCapture", return_value=FakeCapture()):
            events, evidence_count = api.create_job_candidate_events(
                job_id, self.root / "source.mp4", detections,
            )

        self.assertEqual(events, [])
        self.assertEqual(evidence_count, 0)
        payload = json.loads((api.JOBS_DIR / f"{job_id}_candidate_events.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["category_analyses"]["triple_riding"]["status"], "analysis_unavailable")
        self.assertEqual(payload["category_analyses"]["unsafe_pedestrian_crossing"]["status"], "analysis_unavailable")
        self.assertEqual(payload["category_analyses"]["helmetless_riding"]["status"], "analysis_unavailable")

    def test_video_analysis_applies_configured_tracking_confidence_and_image_size(self):
        model_path = self.root / "model.pt"
        model_path.write_bytes(b"test")
        input_path = self.root / "source.mp4"
        input_path.write_bytes(b"test")
        job_id = "123456abcdef"
        api.jobs[job_id] = {"job_id": job_id, "original_filename": "source.mp4"}
        tracking_settings = {"confidence": 0.41, "imgsz": 640, "tracker_config": "test-tracker.yaml"}
        self.patches.enter_context(patch.object(api, "get_model_path", return_value=model_path))
        self.patches.enter_context(patch.object(api, "get_tracking_settings", return_value=tracking_settings))
        self.patches.enter_context(patch.object(api, "get_event_settings", return_value={
            "min_duration_seconds": 3.0,
            "stationary_speed_px_per_second": 5.0,
            "min_detection_confidence": 0.35,
        }))
        self.patches.enter_context(patch.object(api, "create_job_candidate_events", return_value=([], 0)))

        class FakeCapture:
            def isOpened(self):
                return True

            def get(self, property_id):
                return {
                    api.cv2.CAP_PROP_FPS: 30.0,
                    api.cv2.CAP_PROP_FRAME_WIDTH: 16,
                    api.cv2.CAP_PROP_FRAME_HEIGHT: 12,
                    api.cv2.CAP_PROP_FRAME_COUNT: 1,
                }.get(property_id, 0)

            def release(self):
                pass

        class FakeWriter:
            def isOpened(self):
                return True

            def write(self, _frame):
                pass

            def release(self):
                pass

        model = SimpleNamespace(names={}, predictor=object(), track=unittest.mock.Mock(return_value=[
            SimpleNamespace(orig_img=__import__("numpy").zeros((12, 16, 3), dtype="uint8"), boxes=None, names={}),
        ]))
        fake_ultralytics = SimpleNamespace(YOLO=lambda _path: model)
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        self.patches.enter_context(patch.dict(sys.modules, {"ultralytics": fake_ultralytics, "torch": fake_torch}))
        self.patches.enter_context(patch.object(api.cv2, "VideoCapture", return_value=FakeCapture()))
        self.patches.enter_context(patch.object(api.cv2, "VideoWriter", return_value=FakeWriter()))
        self.patches.enter_context(patch.object(api.cv2, "VideoWriter_fourcc", return_value=0))
        self.patches.enter_context(patch.object(api.imageio_ffmpeg, "get_ffmpeg_exe", return_value="ffmpeg"))

        def fake_ffmpeg(command, **_kwargs):
            Path(command[-1]).write_bytes(b"encoded")

        self.patches.enter_context(patch.object(api.subprocess, "run", side_effect=fake_ffmpeg))
        self.patches.enter_context(patch.object(api, "save_persisted_video_job", return_value=None))
        api.JOBS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = self.root / "annotated.mp4"

        api.run_video_analysis(job_id, input_path, output_path, self.root / "detections.csv")

        self.assertEqual(api.jobs[job_id]["status"], "completed")
        self.assertEqual(model.track.call_args.kwargs["conf"], 0.41)
        self.assertEqual(model.track.call_args.kwargs["imgsz"], 640)
        self.assertEqual(model.track.call_args.kwargs["tracker"], "test-tracker.yaml")
        self.assertIsNone(model.predictor)
        self.assertEqual(api.jobs[job_id]["model_checkpoint_path"], str(model_path.resolve()))
        self.assertEqual(api.jobs[job_id]["model_class_names"], {})
        self.assertEqual(api.jobs[job_id]["torch_version"], "unknown")
        self.assertTrue(output_path.is_file())

    def test_model_cache_reuses_weights_and_reloads_when_checkpoint_changes(self):
        checkpoint = self.root / "model.pt"
        checkpoint.write_bytes(b"first checkpoint")
        loaded = []
        fake_ultralytics = SimpleNamespace(YOLO=lambda path: loaded.append((path, object())) or loaded[-1][1])

        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            first = api.load_cached_yolo_model(checkpoint, "road-scene")
            second = api.load_cached_yolo_model(checkpoint, "road-scene")
            self.assertIs(first, second)
            self.assertEqual(len(loaded), 1)

            checkpoint.write_bytes(b"replacement checkpoint with a different size")
            replacement = api.load_cached_yolo_model(checkpoint, "road-scene")

        self.assertIsNot(first, replacement)
        self.assertEqual(len(loaded), 2)

    def test_event_loader_ignores_untracked_negative_and_invalid_ids(self):
        source = self.root / "detections.csv"
        fields = ["frame", "timestamp_seconds", "track_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
        with source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for track_id in ("-1", "not-a-track", "12.0"):
                writer.writerow({"frame": 0, "timestamp_seconds": 0, "track_id": track_id, "class_name": "car", "confidence": 0.8, "x1": 1, "y1": 1, "x2": 10, "y2": 10})
        tracks = load_tracks(source)
        self.assertEqual(list(tracks), ["12"])

    def test_wrong_way_candidates_require_roi_and_opposite_calibrated_movement(self):
        def point(i, x, y=50, confidence=0.8):
            return {"frame": i, "timestamp_seconds": float(i), "cx": x, "cy": y,
                    "confidence": confidence, "class_name": "car"}
        calibration = {"roi_polygon": [{"x": 0.1, "y": 0.1}, {"x": 0.9, "y": 0.1}, {"x": 0.9, "y": 0.9}, {"x": 0.1, "y": 0.9}],
                       "direction_start": {"x": 0.2, "y": 0.5}, "direction_end": {"x": 0.8, "y": 0.5}}
        opposite = [point(i, x, y=500) for i, x in enumerate((850, 700, 550, 400))]
        along = [point(i, x, y=500) for i, x in enumerate((400, 550, 700, 850))]
        low_conf = [point(i, x, y=500, confidence=0.2) for i, x in enumerate((850, 700, 550, 400))]
        outside = [point(i, x, y=950) for i, x in enumerate((850, 700, 550, 400))]
        result = analyze_wrong_way_tracks({"1": opposite, "2": along, "3": low_conf, "4": outside}, calibration, 1000, 1000)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["event_type"], "wrong_way_movement_candidate")
        self.assertEqual(result[0]["status"], "needs_human_review")
        with self.assertRaises(ValueError):
            analyze_wrong_way_tracks({"1": opposite}, {**calibration, "roi_polygon": []}, 1000, 1000)

    def test_video_job_events_merge_latest_directional_report_and_can_be_reviewed(self):
        self.root.mkdir(exist_ok=True)
        api.JOBS_DIR.mkdir(parents=True)
        job_id = "123456abcdef"
        (api.JOBS_DIR / f"{job_id}_candidate_events.json").write_text(json.dumps({"events": [{"event_id": "base", "event_type": "movement_observation", "status": "observation"}]}), encoding="utf-8")
        (api.JOBS_DIR / f"{job_id}_directional_events_20261030T100000000000Z.json").write_text(json.dumps({"events": [{"event_id": "directional", "event_type": "wrong_way_movement_candidate", "status": "needs_human_review"}]}), encoding="utf-8")
        result = api.get_video_job_events(job_id)
        self.assertEqual({event["event_id"] for event in result["events"]}, {"base", "directional"})
        self.assertEqual(result["event_type_counts"]["wrong_way_movement_candidate"], 1)
        self.assertEqual(api.load_reviewable_event("directional", job_id)["event_type"], "wrong_way_movement_candidate")

    def test_directional_calibration_rejects_malformed_points(self):
        api.JOBS_DIR.mkdir(parents=True)
        (api.JOBS_DIR / "123456abcdef_detections.csv").write_text("frame,timestamp_seconds,track_id,class_name,confidence,x1,y1,x2,y2\n", encoding="utf-8")
        with self.assertRaises(api.HTTPException) as error:
            api.create_directional_candidates("123456abcdef", {"roi_polygon": [], "direction_start": {}, "direction_end": {}})
        self.assertEqual(error.exception.status_code, 422)

    def test_directional_calibration_saves_timestamped_candidates_and_updates_job(self):
        job_id = "123456abcdef"
        api.JOBS_DIR.mkdir(parents=True)
        api.JOB_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        api.UPLOADS_DIR.mkdir(parents=True)
        (api.JOBS_DIR / f"{job_id}_detections.csv").write_text(
            "frame,timestamp_seconds,track_id,class_name,confidence,x1,y1,x2,y2\n"
            "0,0,1,car,0.9,840,480,860,520\n"
            "1,1,1,car,0.9,690,480,710,520\n"
            "2,2,1,car,0.9,540,480,560,520\n"
            "3,3,1,car,0.9,390,480,410,520\n", encoding="utf-8")
        (api.JOBS_DIR / f"{job_id}_candidate_events.json").write_text(json.dumps({"events": [], "event_type_counts": {}}), encoding="utf-8")
        (api.UPLOADS_DIR / "source.mp4").write_bytes(b"placeholder")
        api.jobs[job_id] = {"job_id": job_id, "status": "completed", "width": 1000, "height": 1000, "input_video_url": "/uploaded-videos/source.mp4"}
        calibration = {"roi_polygon": [{"x": 0.1, "y": 0.1}, {"x": 0.9, "y": 0.1}, {"x": 0.9, "y": 0.9}, {"x": 0.1, "y": 0.9}], "direction_start": {"x": 0.2, "y": 0.5}, "direction_end": {"x": 0.8, "y": 0.5}}

        class FakeCapture:
            def isOpened(self): return True
            def set(self, *_args): return True
            def read(self): return True, __import__("numpy").zeros((1000, 1000, 3), dtype="uint8")
            def release(self): pass

        with patch.object(api.cv2, "VideoCapture", return_value=FakeCapture()), patch.object(api.cv2, "imwrite", return_value=True):
            result = api.create_directional_candidates(job_id, calibration)

        self.assertEqual(result["total"], 1)
        self.assertEqual(len(result["events"]), 1)
        self.assertTrue(list(api.JOBS_DIR.glob(f"{job_id}_directional_events_*.json")))
        self.assertEqual(api.jobs[job_id]["directional_candidate_count"], 1)

    def test_health_and_upload_readiness_follow_model_availability(self):
        checkpoint = self.root / "best.pt"
        checkpoint.write_bytes(b"model")
        with patch.object(api, "get_model_path", return_value=checkpoint):
            self.assertTrue(api.health()["upload_enabled"])
        with patch.object(api, "get_model_path", return_value=self.root / "missing.pt"):
            self.assertFalse(api.health()["upload_enabled"])

    def test_upload_fails_before_saving_video_when_model_is_missing(self):
        self.patches.enter_context(patch.object(api, "UPLOADS_DIR", self.root / "uploads"))
        self.patches.enter_context(patch.object(api, "get_model_path", return_value=self.root / "missing.pt"))
        upload = UploadFile(filename="road.mp4", file=io.BytesIO(b"video"))
        with self.assertRaises(api.HTTPException) as error:
            asyncio.run(api.upload_video(api.BackgroundTasks(), upload))
        self.assertEqual(error.exception.status_code, 503)
        self.assertFalse((self.root / "uploads").exists())

    def test_upload_rejects_unreadable_video_and_removes_temporary_file(self):
        self.patches.enter_context(patch.object(api, "UPLOADS_DIR", self.root / "uploads"))
        self.patches.enter_context(patch.object(
            api, "get_analysis_readiness",
            return_value={"model_available": True, "upload_enabled": True, "configuration_error": None},
        ))
        self.patches.enter_context(patch.object(api, "is_readable_video", return_value=False))
        upload = UploadFile(filename="road.mp4", file=io.BytesIO(b"not a video"))

        with self.assertRaises(api.HTTPException) as error:
            asyncio.run(api.upload_video(api.BackgroundTasks(), upload))

        self.assertEqual(error.exception.status_code, 400)
        self.assertEqual(list((self.root / "uploads").glob("*")), [])
        self.assertEqual(api.jobs, {})

    def test_upload_accepts_validated_video_and_uses_safe_generated_filename(self):
        checkpoint = self.root / "best.pt"
        checkpoint.write_bytes(b"model")
        self.patches.enter_context(patch.object(api, "UPLOADS_DIR", self.root / "uploads"))
        self.patches.enter_context(patch.object(
            api, "get_analysis_readiness",
            return_value={"model_available": True, "upload_enabled": True, "configuration_error": None},
        ))
        self.patches.enter_context(patch.object(api, "is_readable_video", return_value=True))
        upload = UploadFile(filename="..\\unsafe name.mp4", file=io.BytesIO(b"valid video bytes"))
        background_tasks = api.BackgroundTasks()

        job = asyncio.run(api.upload_video(background_tasks, upload))

        saved_path = next((self.root / "uploads").glob("*.mp4"))
        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["original_filename"], "unsafe name.mp4")
        self.assertEqual(saved_path.name, f"{job['job_id']}_unsafe_name.mp4")
        self.assertEqual(len(background_tasks.tasks), 1)
        self.assertIs(background_tasks.tasks[0].func, api.enqueue_video_analysis)

    def test_missing_source_video_fails_job_without_loading_models(self):
        job_id = "123456abcdef"
        api.jobs[job_id] = {"job_id": job_id, "status": "queued"}
        missing_video = self.root / "missing.mp4"

        api.run_video_analysis(job_id, missing_video, self.root / "annotated.mp4", self.root / "detections.csv")

        self.assertEqual(api.jobs[job_id]["status"], "failed")
        self.assertEqual(api.jobs[job_id]["error"], "Video processing failed. Check backend logs for details.")

    def test_invalid_analysis_settings_disable_upload_and_fail_early(self):
        checkpoint = self.root / "best.pt"
        checkpoint.write_bytes(b"model")
        self.patches.enter_context(patch.object(api, "get_model_path", return_value=checkpoint))
        self.patches.enter_context(patch.object(api, "UPLOADS_DIR", self.root / "uploads"))
        self.patches.enter_context(patch.object(
            api,
            "get_backend_setting",
            side_effect=lambda name: "bad-value" if name == "ROAD_SENTINEL_EVENT_MIN_DURATION_SECONDS" else "",
        ))
        readiness = api.get_analysis_readiness()
        self.assertTrue(readiness["model_available"])
        self.assertFalse(readiness["upload_enabled"])
        self.assertIn("ROAD_SENTINEL_EVENT_MIN_DURATION_SECONDS", readiness["configuration_error"])
        upload = UploadFile(filename="road.mp4", file=io.BytesIO(b"video"))
        with self.assertRaises(api.HTTPException) as error:
            asyncio.run(api.upload_video(api.BackgroundTasks(), upload))
        self.assertEqual(error.exception.status_code, 503)
        self.assertFalse((self.root / "uploads").exists())

    def test_q_and_a_filters_unknown_citations_and_scopes_session_to_job(self):
        self.patches.enter_context(patch.object(api, "get_video_job_events", return_value={
            "events": [{
                "event_id": "EVT-1", "event_type": "stationary_object_candidate",
                "status": "needs_human_review", "start_seconds": 5.5,
            }],
        }))
        self.mock_provider({
            "answer": "A stationary candidate was recorded at 00:05.5.",
            "citations": [
                {"event_id": "EVT-1", "reason": "The record labels this a stationary candidate."},
                {"event_id": "EVT-UNKNOWN", "reason": "Must be discarded."},
            ],
            "insufficient_data": False,
        })

        result = api.ask_job_question("123456abcdef", "What was flagged?")

        self.assertEqual([item["event_id"] for item in result["citations"]], ["EVT-1"])
        history = api.get_job_qa_messages("123456abcdef", result["session_id"])
        self.assertEqual(len(history), 2)
        self.assertEqual(history[-1]["source_event_count"], 1)
        with self.assertRaises(api.HTTPException) as error:
            api.get_job_qa_messages("abcdef123456", result["session_id"])
        self.assertEqual(error.exception.status_code, 404)

    def test_insufficient_data_answers_do_not_keep_citations(self):
        self.patches.enter_context(patch.object(api, "get_video_job_events", return_value={
            "events": [{"event_id": "EVT-1", "event_type": "movement_observation", "status": "observation"}],
        }))
        self.mock_provider({
            "answer": "The event record does not include the camera angle.",
            "citations": [{"event_id": "EVT-1", "reason": "Contradicts insufficient result."}],
            "insufficient_data": True,
        })

        result = api.ask_job_question("123456abcdef", "What was the camera angle?")

        self.assertTrue(result["insufficient_data"])
        self.assertEqual(result["citations"], [])
        self.assertTrue(api.get_job_qa_messages("123456abcdef", result["session_id"])[-1]["insufficient_data"])

    def test_event_explanation_uses_only_structured_record_and_saves_json(self):
        self.patches.enter_context(patch.object(api, "get_video_job_events", return_value={
            "events": [{
                "event_id": "EVT-7", "event_type": "stationary_object_candidate",
                "status": "needs_human_review", "start_seconds": 12.4,
                "evidence_url": "/evidence/frame.jpg", "notes": "Review source footage.",
            }],
        }))
        client = self.mock_provider({
            "explanation": "The record flags a stationary object for review.",
            "supporting_facts": ["Its status is needs human review."],
            "uncertainties": ["The record cannot establish a traffic violation."],
            "reviewer_guidance": "Inspect the original footage.",
        })

        report = api.generate_job_event_ai_explanation("123456abcdef", "EVT-7")

        request_body = client.post.call_args.kwargs["json"]
        self.assertNotIn("evidence_url", request_body["input"])
        self.assertNotIn("frame.jpg", request_body["input"])
        saved_path = self.root / "video_jobs" / report["download_url"].rsplit("/", 1)[-1]
        self.assertTrue(saved_path.is_file())
        self.assertEqual(json.loads(saved_path.read_text(encoding="utf-8"))["event_id"], "EVT-7")

    def test_event_explanation_rejects_event_from_another_job(self):
        self.patches.enter_context(patch.object(api, "get_video_job_events", return_value={"events": []}))
        with self.assertRaises(api.HTTPException) as error:
            api.generate_job_event_ai_explanation("123456abcdef", "EVT-404")
        self.assertEqual(error.exception.status_code, 404)

    def test_video_job_report_counts_review_states_without_promoting_candidates(self):
        job_id = "123456abcdef"
        api.jobs[job_id] = {
            "job_id": job_id,
            "status": "completed",
            "original_filename": "traffic.mp4",
            "created_at": "2026-10-31T10:00:00+00:00",
        }
        events = [
            {"event_id": "confirmed", "status": "confirmed_by_reviewer", "start_seconds": 8},
            {"event_id": "pending", "status": "needs_human_review", "start_seconds": 4},
            {"event_id": "candidate", "status": "candidate_violation", "start_seconds": 5},
            {"event_id": "rejected", "status": "rejected_by_reviewer", "start_seconds": 6},
            {"event_id": "observation", "status": "observation", "start_seconds": 7},
            {"event_id": "legacy", "status": "observation_not_violation", "start_seconds": 9},
            {"event_id": "unknown", "status": "future_status", "start_seconds": 10},
        ]
        self.patches.enter_context(patch.object(api, "get_video_job_events", return_value={"events": events}))

        report = api.get_video_job_report(job_id)

        self.assertEqual(report["summary"]["total_events"], 7)
        self.assertEqual(report["summary"]["counts"], {
            "confirmed_by_reviewer": 1,
            "needs_human_review": 2,
            "rejected_by_reviewer": 1,
            "observation": 2,
            "unclassified": 1,
        })
        self.assertEqual(report["summary"]["confirmed_message"], "1 event(s) explicitly confirmed by a reviewer.")
        self.assertEqual([event["event_id"] for event in report["events"]], [
            "pending", "candidate", "rejected", "observation", "confirmed", "legacy", "unknown",
        ])

    def test_ai_summary_instructions_distinguish_observations_and_record_counts(self):
        events = [
            {"event_id": "observation", "event_type": "movement_observation", "status": "observation"},
            {"event_id": "pending", "event_type": "stationary_object_candidate", "status": "needs_human_review"},
        ]
        self.patches.enter_context(patch.object(
            api, "get_video_job_events", return_value={"uploaded_filename": "traffic.mp4", "events": events},
        ))
        client = self.mock_provider({
            "executive_summary": "Two event records were supplied for review.",
            "observed_patterns": [],
            "events_to_review": [],
            "reviewed_events": [],
            "limitations": [],
        })
        api.JOBS_DIR.mkdir(parents=True, exist_ok=True)

        api.generate_job_ai_summary("123456abcdef")

        request_body = client.post.call_args.kwargs["json"]
        instructions = request_body["instructions"]
        self.assertIn("untested movement records", instructions)
        self.assertIn("Never say that any track followed normal or standard movement", instructions)
        self.assertIn("total_candidate_events as tracked vehicles", instructions)
        self.assertIn("quote counts only from the supplied count fields", instructions)
        context = json.loads(request_body["input"])
        self.assertEqual(context["total_candidate_events"], 2)
        self.assertEqual(context["review_status_counts"], {"observation": 1, "needs_human_review": 1})

    def test_video_job_report_empty_events_and_missing_evidence_are_supported(self):
        job_id = "123456abcdef"
        api.jobs[job_id] = {"job_id": job_id, "status": "completed", "original_filename": "empty.mp4"}
        self.patches.enter_context(patch.object(
            api, "get_video_job_events", side_effect=api.HTTPException(status_code=404, detail="No events"),
        ))

        report = api.get_video_job_report(job_id)

        self.assertEqual(report["events"], [])
        self.assertEqual(report["summary"]["total_events"], 0)
        self.assertEqual(report["summary"]["counts"], {
            "confirmed_by_reviewer": 0,
            "needs_human_review": 0,
            "rejected_by_reviewer": 0,
            "observation": 0,
            "unclassified": 0,
        })
        self.assertEqual(
            report["summary"]["confirmed_message"],
            "No reviewer-confirmed violations recorded.",
        )

    def test_video_job_report_accepts_events_without_evidence(self):
        job_id = "123456abcdef"
        api.jobs[job_id] = {"job_id": job_id, "status": "completed"}
        event = {
            "event_id": "EVT-NO-EVIDENCE",
            "event_type": "movement_observation",
            "status": "observation",
            "start_seconds": 3.0,
        }
        self.patches.enter_context(patch.object(api, "get_video_job_events", return_value={"events": [event]}))

        report = api.get_video_job_report(job_id)

        self.assertEqual(len(report["events"]), 1)
        self.assertEqual(report["events"][0]["event_id"], "EVT-NO-EVIDENCE")
        self.assertNotIn("evidence_url", report["events"][0])

    def test_video_job_report_distinguishes_category_states_and_parking_events(self):
        job_id = "123456abcdef"
        api.jobs[job_id] = {"job_id": job_id, "status": "completed"}
        api.JOBS_DIR.mkdir(parents=True, exist_ok=True)
        parking_event = {
            "event_id": "PARK-1",
            "event_type": "parking_zone_candidate",
            "status": "needs_human_review",
        }
        (api.JOBS_DIR / f"{job_id}_parking_events_20261104T100000Z.json").write_text(
            json.dumps({"events": [parking_event]}), encoding="utf-8",
        )
        (api.JOBS_DIR / f"{job_id}_directional_events_20261104T100000Z.json").write_text(
            "{invalid", encoding="utf-8",
        )
        self.patches.enter_context(patch.object(
            api, "get_video_job_events", return_value={"events": [parking_event]},
        ))

        report = api.get_video_job_report(job_id)

        categories = {category["id"]: category for category in report["violation_categories"]}
        self.assertEqual(len(categories), 6)
        self.assertEqual(set(categories), {
            "helmetless_riding", "triple_riding", "illegal_parking",
            "overloaded_goods_vehicle", "unsafe_pedestrian_crossing", "wrong_way_movement",
        })
        self.assertEqual(categories["illegal_parking"]["status"], "events_detected")
        self.assertEqual(categories["illegal_parking"]["event_count"], 1)
        self.assertEqual(categories["illegal_parking"]["event_ids"], ["PARK-1"])
        self.assertEqual(categories["wrong_way_movement"]["status"], "analysis_failed")
        self.assertEqual(categories["helmetless_riding"]["status"], "analysis_unavailable")
        self.assertEqual(categories["overloaded_goods_vehicle"]["status"], "analysis_unavailable")
        self.assertEqual(categories["unsafe_pedestrian_crossing"]["status"], "analysis_unavailable")

    def test_video_job_report_marks_configured_empty_analyzers_and_unconfigured_analyzers(self):
        job_id = "123456abcdef"
        api.jobs[job_id] = {"job_id": job_id, "status": "completed"}
        api.JOBS_DIR.mkdir(parents=True, exist_ok=True)
        (api.JOBS_DIR / f"{job_id}_parking_events_20261104T100000Z.json").write_text(
            json.dumps({"events": []}), encoding="utf-8",
        )
        self.patches.enter_context(patch.object(api, "get_video_job_events", return_value={"events": []}))

        report = api.get_video_job_report(job_id)

        categories = {category["id"]: category for category in report["violation_categories"]}
        self.assertEqual(categories["illegal_parking"]["status"], "no_events")
        self.assertEqual(categories["illegal_parking"]["event_count"], 0)
        self.assertEqual(categories["wrong_way_movement"]["status"], "not_configured")

    def test_video_job_report_uses_saved_category_analysis_outcomes(self):
        job_id = "123456abcdef"
        api.jobs[job_id] = {"job_id": job_id, "status": "completed"}
        self.patches.enter_context(patch.object(
            api,
            "get_video_job_events",
            return_value={
                "events": [],
                "category_analyses": {
                    "triple_riding": {"status": "no_events", "message": "Rider and motorcycle detections were analyzed."},
                    "unsafe_pedestrian_crossing": {"status": "analysis_unavailable", "message": "Required detections were not both available."},
                },
            },
        ))

        report = api.get_video_job_report(job_id)

        categories = {category["id"]: category for category in report["violation_categories"]}
        self.assertEqual(categories["triple_riding"]["status"], "no_events")
        self.assertEqual(categories["unsafe_pedestrian_crossing"]["status"], "analysis_unavailable")
        self.assertEqual(categories["triple_riding"]["message"], "Rider and motorcycle detections were analyzed.")

    def test_video_job_report_marks_in_progress_categories_and_event_details(self):
        job_id = "123456abcdef"
        api.jobs[job_id] = {"job_id": job_id, "status": "processing"}
        self.patches.enter_context(patch.object(
            api, "get_video_job_events", return_value={"events": []},
        ))

        report = api.get_video_job_report(job_id)

        self.assertEqual(len(report["violation_categories"]), 6)
        self.assertTrue(all(
            category["status"] == "processing_incomplete"
            for category in report["violation_categories"]
        ))

        event = {
            "event_id": "HELMET-1",
            "event_type": "helmetless_riding_candidate",
            "status": "needs_human_review",
            "timestamp_seconds": 4.2,
            "track_id": 8,
            "class_name": "rider",
            "confidence": 0.81,
            "notes": "Head region may be occluded.",
            "evidence_url": "/job-evidence/123/helmet.jpg",
        }
        self.patches.enter_context(patch.object(
            api, "get_video_job_events", return_value={"events": [event]},
        ))
        api.jobs[job_id]["status"] = "completed"

        report = api.get_video_job_report(job_id)
        helmet = next(item for item in report["violation_categories"] if item["id"] == "helmetless_riding")
        self.assertEqual(helmet["status"], "events_detected")
        self.assertEqual(helmet["events"][0]["category"], "helmetless_riding")
        self.assertEqual(helmet["events"][0]["timestamp"], 4.2)
        self.assertEqual(helmet["events"][0]["object_class"], "rider")
        self.assertEqual(helmet["events"][0]["review_status"], "needs_human_review")
        self.assertEqual(helmet["events"][0]["uncertainty"], "Head region may be occluded.")
        self.assertEqual(helmet["events"][0]["evidence_path"], "/job-evidence/123/helmet.jpg")

    def test_video_job_report_survives_corrupt_category_report(self):
        job_id = "123456abcdef"
        api.jobs[job_id] = {"job_id": job_id, "status": "completed"}
        api.JOBS_DIR.mkdir(parents=True, exist_ok=True)
        (api.JOBS_DIR / f"{job_id}_candidate_events.json").write_text(
            json.dumps({"events": [{"event_id": "base", "event_type": "movement_observation", "status": "observation"}]}),
            encoding="utf-8",
        )
        (api.JOBS_DIR / f"{job_id}_directional_events_20261104T100000Z.json").write_text(
            "{invalid", encoding="utf-8",
        )

        report = api.get_video_job_report(job_id)

        categories = {category["id"]: category for category in report["violation_categories"]}
        self.assertEqual(report["events"][0]["event_id"], "base")
        self.assertEqual(categories["wrong_way_movement"]["status"], "analysis_failed")
        self.assertEqual(categories["illegal_parking"]["status"], "not_configured")
        self.assertEqual(report["analysis_errors"], [])

    def test_video_job_report_survives_corrupt_base_event_data(self):
        job_id = "123456abcdef"
        api.jobs[job_id] = {"job_id": job_id, "status": "completed"}
        api.JOBS_DIR.mkdir(parents=True, exist_ok=True)
        (api.JOBS_DIR / f"{job_id}_candidate_events.json").write_text(
            "{invalid", encoding="utf-8",
        )

        report = api.get_video_job_report(job_id)

        categories = {category["id"]: category for category in report["violation_categories"]}
        self.assertEqual(report["events"], [])
        self.assertEqual(categories["triple_riding"]["status"], "analysis_failed")
        self.assertEqual(categories["unsafe_pedestrian_crossing"]["status"], "analysis_failed")
        self.assertEqual(len(report["analysis_errors"]), 1)

    def test_video_job_report_preserves_evidence_and_reviewer_history_when_present(self):
        job_id = "123456abcdef"
        api.jobs[job_id] = {"job_id": job_id, "status": "completed"}
        event = {
            "event_id": "EVT-22",
            "event_type": "wrong_way_movement_candidate",
            "status": "needs_human_review",
            "start_seconds": 12.5,
            "evidence_url": "/api/v1/evidence/frame-22.jpg",
            "reviewer_note": "Check the original footage.",
            "review_history": [{"status": "needs_human_review", "reviewer": "reviewer-1"}],
        }
        self.patches.enter_context(patch.object(api, "get_video_job_events", return_value={"events": [event]}))

        report = api.get_video_job_report(job_id)

        self.assertEqual(report["events"][0]["evidence_url"], event["evidence_url"])
        self.assertEqual(report["events"][0]["reviewer_note"], event["reviewer_note"])
        self.assertEqual(report["events"][0]["review_history"], event["review_history"])

    def test_video_job_report_rejects_invalid_or_missing_jobs(self):
        with self.assertRaises(api.HTTPException) as invalid_error:
            api.get_video_job_report("not-a-valid-id")
        self.assertEqual(invalid_error.exception.status_code, 404)

        with self.assertRaises(api.HTTPException) as missing_error:
            api.get_video_job_report("123456abcdef")
        self.assertEqual(missing_error.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
