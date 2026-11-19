# RoadSentinel

**RoadSentinel is a local video-review prototype for road-scene detection and traffic-risk auditing.** It accepts dashcam or CCTV video, runs a configured YOLO detector with BoT-SORT tracking, and presents annotated playback, timestamped candidate events, evidence, and review-aware reports.

> **Interpretation:** A detector output or analyzer candidate is not a confirmed traffic violation. “No events” means no candidate met configured rules; it does not prove that the video contains no violation. Human review is required.

## Contents

- [Problem and objectives](#problem-and-objectives)
- [Features and category coverage](#features-and-category-coverage)
- [Architecture and project structure](#architecture-and-project-structure)
- [Dataset and model](#dataset-and-model)
- [Install and run on Windows](#install-and-run-on-windows)
- [Video-analysis workflow and reports](#video-analysis-workflow-and-reports)
- [API overview](#api-overview)
- [Configuration](#configuration)
- [Testing and verified results](#testing-and-verified-results)
- [Limitations and future work](#limitations-and-future-work)
- [Submission and demo documents](#submission-and-demo-documents)

## Problem and objectives

Reviewing long road-camera recordings manually can be time-consuming and inconsistent. RoadSentinel explores a local, evidence-first workflow that helps a reviewer locate road-scene objects and candidate events without presenting automated output as a legal conclusion.

Project objectives:

1. Detect and track supported road-scene objects in uploaded video.
2. Produce annotated playback and timestamped review evidence.
3. Apply configurable or experimental rules to selected road-safety categories.
4. Present category status—including unavailable, unconfigured, failed, and no-candidate outcomes—alongside events.
5. Let a reviewer inspect evidence and record a decision.

## Features and category coverage

| Category | Implemented behavior | Status and caveat |
| --- | --- | --- |
| Helmetless riding | Optional specialist-model analysis associates explicit helmet-state detections with riders over repeated observations and requires a spatially matching primary-detector `rider` box. | Unavailable unless a compatible helmet-specific checkpoint is configured. Candidates failing primary-rider corroboration are rejected; this does not establish helmet-model accuracy. |
| Triple riding | Experimental spatial and temporal association of rider detections with one motorcycle. | Heuristic candidate only; nearby people, overlap, and occlusion can cause errors. |
| Restricted-zone parking | Parking analyzer uses configured restricted-area geometry and stationary-duration criteria. | Requires user calibration. A stationary vehicle is not necessarily illegally parked. |
| Goods-vehicle load risk | Report category and limitation are represented. | Analysis unavailable: no validated cargo-risk detector or load measurement is configured. Truck detection does not establish overloading. |
| Unsafe pedestrian interaction | Experimental tracked pedestrian/vehicle proximity heuristic. It requires a moving person track, decreasing image-space separation, and sustained proximity; reports identify the person as the subject and vehicle as context. | Candidate risk signal only. The motion threshold is 0.4 median person-box heights, and image-space separation is not physical distance, crossing intent, or collision probability. |
| Wrong-way movement | Directional analyzer compares tracked movement with a user-configured road area and expected direction. | Requires calibration; perspective and track noise may affect results. |

The combined report always includes the six categories. Its status values distinguish processing-incomplete, events-detected, no-events, analysis-failed, analysis-unavailable, and not-configured states.

Other implemented workflow features include video upload validation, background job progress, annotated video, detection exports, event evidence, event review history, and optional AI text explanations over structured event records. Optional AI features require backend credentials; source video and evidence frames are not sent to the AI provider by those explanation features.

## Architecture and project structure

```text
RoadSentinel/
├── api.py                         FastAPI routes and video-job orchestration
├── tools/                          Detection, tracking, and event-analysis helpers
├── tests/                          Backend and analyzer tests
├── frontend/                       React + TypeScript + Vite application
├── samples/                        Local demo/test clips
├── docs/                           Architecture, academic, demo, and checklist docs
├── outputs/                         Local runtime data; ignored by Git
├── requirements.txt                Python backend dependencies
└── .env.example                    Backend configuration template
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the processing sequence, component responsibilities, persistence, and report semantics.

**Technology:** Python, FastAPI, Uvicorn, OpenCV, Ultralytics YOLO, BoT-SORT, PyTorch, SQLite, JSON/CSV, React, TypeScript, Vite, and `lucide-react`. Exact Python package versions are environment-dependent; see `requirements.txt` and `frontend/package-lock.json`.

## Dataset and model

- The detector was trained/validated using the project’s processed IDD/YOLO dataset. The dataset YAML is maintained outside this repository at `D:\RoadSentinel-Datasets\processed\IDD_YOLO\idd.yaml`; dataset files and dataset provenance/license material are not bundled here.
- The recorded detector class map contains 15 object classes: motorcycle, rider, car, person, autorickshaw, truck, vehicle fallback, bus, traffic sign, animal, traffic light, bicycle, caravan, train, and trailer.
- The backend’s baseline checkpoint is `D:\RoadSentinel-Datasets\training-runs\idd_yolov8n_full\weights\best.pt`. It is not included in this repository. Set `ROAD_SENTINEL_MODEL_PATH` if it is stored elsewhere.
- The IDD model detects object classes, not the six violation labels. Helmet analysis needs a compatible specialist model. No validated goods-vehicle cargo-risk model is configured.
- Saved baseline validation metrics: precision `0.61192`, recall `0.26038`, mAP@0.5 `0.28612`, and mAP@0.5:0.95 `0.18102`. These are dataset validation metrics, not real-world violation accuracy.
- A recorded fine-tune has metrics of `0.63944`, `0.26263`, `0.29318`, and `0.18473` respectively at 11/50 planned epochs. Its training configuration differs; it has not been promoted over the baseline.

Runtime evaluation details, per-class AP values, and interpretation guidance are in [docs/ACADEMIC_PRESENTATION.md](docs/ACADEMIC_PRESENTATION.md). Do not copy model weights or datasets into generated output directories or redistribute external assets without checking their license.

## Install and run on Windows

Prerequisites: Windows, Python with the backend dependencies, a compatible PyTorch installation, and Node.js (20.19+ or 22.12+ recommended). The verified workspace Python is `C:\Projects\RoadSentinel\.venv\Scripts\python.exe`; substitute your actual Python executable if different.

### 1. Configure the backend

From Command Prompt:

```bat
cd /d C:\Users\Admin\Desktop\BTECH\RoadSentinel
copy .env.example .env
```

Set `ROAD_SENTINEL_MODEL_PATH` in `.env` if the baseline checkpoint is not at its default external path. Install dependencies in the intended environment if needed:

```bat
C:\Projects\RoadSentinel\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install a PyTorch build suitable for the machine separately; the project requirements do not pin the CUDA-specific PyTorch wheel.

### 2. Start the backend

The default demo ports `8765` and `5173` are occupied by existing RoadSentinel
processes on the verified demonstration machine. Leave those processes alone
and use the tested alternate ports `8766` (backend) and `5174` (frontend).
The backend CORS allowlist already permits frontend port `5174`.

In the first Command Prompt:

```bat
cd /d C:\Users\Admin\Desktop\BTECH\RoadSentinel
C:\Projects\RoadSentinel\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8766
```

Check `http://127.0.0.1:8766/health`. Interactive API documentation is at `http://127.0.0.1:8766/docs`.

### 3. Start the frontend

In a second Command Prompt. The explicit `--port` overrides the Vite config's
default port; setting the API URL in this terminal ensures requests go to the
backend started above, even if a local `frontend/.env` still names port `8765`.

```bat
cd /d C:\Users\Admin\Desktop\BTECH\RoadSentinel\frontend
npm ci
set "VITE_API_BASE_URL=http://127.0.0.1:8766"
npm run dev -- --port 5174
```

Open `http://127.0.0.1:5174/`. The matching frontend API setting is also in
`frontend/.env.example`. Never put backend secrets in the frontend environment.

## Video-analysis workflow and reports

1. Choose a supported MP4, AVI, MOV, MKV, or WEBM video in the dashboard. Uploads are limited to 1 GB and checked for readable video content.
2. The API queues background analysis. The UI polls job state and shows processing progress.
3. The backend runs YOLO detection and tracking, writes detections and annotated playback, and runs configured/available event analyzers.
4. Candidate evidence and analysis outcomes are saved. The report lists all six category states, events, timestamps, evidence, and uncertainty/review information when available.
5. Select an event to inspect its evidence and seek the video. A reviewer can record a decision; candidates are not auto-confirmed.

See [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md) for the demonstration sequence and category-by-category talking points.

Generated job videos, detections, JSON reports, and evidence are stored under `outputs/video_jobs/`. Job metadata, review decisions, and optional Q&A are stored in SQLite files under `outputs/`. Outputs are local and ignored by Git; back up required submission evidence separately.

## API overview

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Readiness and model/configuration state |
| `GET` | `/api/v1/summary` | Global dashboard summary |
| `GET` | `/api/v1/events` | Paginated/filterable global event records |
| `GET` | `/api/v1/events/{event_id}` | Event details |
| `POST` | `/api/v1/videos/upload` | Validate and queue a video |
| `GET` | `/api/v1/videos/jobs` | List saved/current jobs |
| `GET` | `/api/v1/videos/jobs/{job_id}` | Job status and progress |
| `GET` | `/api/v1/videos/jobs/{job_id}/events` | Job event records and category outcomes |
| `GET` | `/api/v1/videos/jobs/{job_id}/report` | Combined six-category report |
| `POST` | `/api/v1/videos/jobs/{job_id}/parking-analysis` | Analyze with submitted parking calibration |
| `POST` | `/api/v1/videos/jobs/{job_id}/direction-calibration` | Analyze with submitted road/direction calibration |
| `GET` | `/api/v1/videos/jobs/{job_id}/events.csv` or `.json` | Export job events |
| `POST` | `/api/v1/review-decisions` | Save a reviewer decision |

Additional health, evidence, report-export, and optional AI routes are discoverable at `/docs`. The upload response represents a queued job; poll the job endpoint until completed or failed before relying on its report.

## Configuration

Backend settings are read from process environment or the project-root `.env`. Common options:

| Setting | Purpose |
| --- | --- |
| `ROAD_SENTINEL_MODEL_PATH` | Baseline YOLO checkpoint override (absolute or project-relative path) |
| `ROAD_SENTINEL_TRACK_CONFIDENCE` | Detection/tracking confidence threshold |
| `ROAD_SENTINEL_IMGSZ` | YOLO inference image size (640–1280, divisible by 32) |
| `ROAD_SENTINEL_TRACKER_CONFIG` | Built-in `botsort.yaml`/`bytetrack.yaml` or tracker config path |
| `ROAD_SENTINEL_EVENT_MIN_DURATION_SECONDS` | General minimum track duration |
| `ROAD_SENTINEL_EVENT_STATIONARY_SPEED_PX_S` | Image-space stationary heuristic threshold; not real speed |
| `ROAD_SENTINEL_EVENT_MIN_CONFIDENCE` | Event-summary detection threshold |
| `ROAD_SENTINEL_HELMET_MODEL_PATH` | Specialist helmet checkpoint; `disabled` skips it |
| `ROAD_SENTINEL_HELMET_*` | Helmet inference stride and repeated-observation thresholds |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | Optional AI explanation/Q&A provider settings |
| `OPENAI_TIMEOUT_SECONDS` | Optional provider timeout |

See [.env.example](.env.example) for the full backend template and [frontend/.env.example](frontend/.env.example) for the frontend API URL. Never commit `.env` or put an AI key in browser code.

## Testing and verified results

From the repository root:

```bat
C:\Projects\RoadSentinel\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For frontend type-check and production build:

```bat
cd /d C:\Users\Admin\Desktop\BTECH\RoadSentinel\frontend
npm run build
```

The current validation pass completed **58 backend unit tests** and the frontend TypeScript/production build. The saved detections for `13020022_3840_2160_30fps.mp4` were replayed through the updated candidate analyzers: helmet candidates were retained only when corroborated by the primary detector's `rider` class, and pedestrian-interaction candidates were reduced from 8 to 2 while identifying the person as the road user. The older `road_video.mp4` replay reduced helmet candidates from 5 to 0 and pedestrian-interaction candidates from 89 to 60. These replay counts are not ground-truth precision/recall measurements; both categories remain experimental and require human review. One separate recorded `correct video.mp4` run processed 1,022/1,022 frames, with 3,902 detection rows, 139 unique track IDs, 22 raw candidate-event records, and 22 evidence images. These are pipeline outputs, not violation accuracy metrics.

## Limitations and future work

**Known limitations**

- Detection and tracking can miss, misclassify, or switch object identities, especially under distance, occlusion, or poor image quality.
- Helmet status is unavailable without a compatible specialist checkpoint and has no independent accuracy benchmark.
- Triple-riding and pedestrian-interaction rules are experimental heuristics.
- Parking and wrong-way analysis require camera-specific calibration.
- Cargo-risk/overloading analysis is unavailable; no weight or legal determination is made.
- Image-space movement and separation are not calibrated distance, speed, time-to-collision, or collision probability.
- Processing is local and serialized; interrupted jobs are marked failed and are not automatically resumed.
- Reported validation metrics are from the dataset validation split. No independently annotated real-world violation evaluation or controlled false-positive/false-negative study is available.

**Future improvements** include collecting licensed, representative, annotated violation-specific datasets; validating each analyzer against expert-reviewed ground truth; adding camera calibration where physical measurements are needed; evaluating dedicated helmet/cargo models; and measuring latency and accuracy across varied cameras and weather.

## Submission and demo documents

- [Architecture and processing design](docs/ARCHITECTURE.md)
- [Academic presentation material](docs/ACADEMIC_PRESENTATION.md)
- [Final presentation and viva preparation guide](docs/VIVA_PREPARATION_GUIDE.md)
- [Demo workflow and troubleshooting](docs/DEMO_GUIDE.md)
- [Submission checklist](docs/SUBMISSION_CHECKLIST.md)
- [Prior repository cleanup record](docs/cleanup-log.md)
