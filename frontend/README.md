# RoadSentinel Frontend

React + Vite + TypeScript dashboard for the existing local FastAPI backend.

## Requirements
- Node.js 20.19+ or 22.12+ recommended
- Existing RoadSentinel FastAPI server running at `http://127.0.0.1:8766`

## Run on Windows
Open a new terminal (leave the API terminal running):

```bat
cd /d C:\Users\Admin\Desktop\BTECH\RoadSentinel\frontend
npm install
copy .env.example .env
set "VITE_API_BASE_URL=http://127.0.0.1:8766"
npm run dev -- --port 5174
```

Open `http://127.0.0.1:5174`.

If your API uses another address, edit `.env` and set `VITE_API_BASE_URL`.

## What works
- Live health and summary checks, event list and event details
- Event type/status filters, pagination, and evidence image previews
- Video upload and background YOLO/BoT-SORT processing
- Annotated video playback, detection CSV and JSON summary downloads
- Per-job model checkpoint/runtime provenance and the detector's class-ID map
- Uploaded-video event gallery and timestamp timeline that seeks playback and opens evidence
- Optional wrong-way movement candidates after drawing the road area and expected direction on the video frame
- Combined video report with status and event evidence for all six categories, including calibrated parking candidates
- Reviewer decisions, notes, and decision history
- Event report CSV download
- Responsive graphite dashboard layout

## Current limitations
Upload processing creates detections, tracks, an annotated video, review-only candidate events, and frame
evidence linked to those events. Helmetless-riding candidates require the separate, explicitly helmet-trained
checkpoint configured in the backend; model scores are not calibrated violation probabilities and require
human review.
Video job status and metadata are saved in `outputs\video_jobs.sqlite3`, and the dashboard lists recent
analyses after restart. A job interrupted during processing is marked failed; processing does not resume
automatically, so upload that video again to retry. Generated files remain in `outputs\video_jobs`.
Event records are observations or candidates, not automatically confirmed violations.
The human-readable video report includes only category-specific safety candidates and explicitly shows the analysis
status for each of the six categories. Generic movement observations are not presented as violations in that report;
they remain available in the global event archive and raw CSV/JSON exports.
Wrong-way and parking candidates remain disabled until a reviewer configures their respective calibration. Triple-riding candidates require three distinct rider boxes associated with one motorcycle for at least five analyzed frames and 0.75 seconds; duplicate-box suppression reduces false positives, but this remains an experimental image-space heuristic. Pedestrian/vehicle close-approach checks are also experimental and require suitable tracked classes. Helmet analysis is unavailable if its specialist checkpoint is absent or disabled; the general IDD model does not detect helmet state. Goods-vehicle-load-risk analysis remains unavailable. The report distinguishes unavailable and unconfigured analyses from a configured analysis that found no candidates.

## Backend workflow checks
From the project root, run `python -m unittest discover -s tests -v` with the same Python environment
used to start FastAPI. These tests use mocked provider responses and temporary SQLite files; they do not
need a video, trained weights, GPU, or OpenAI API key.

The dashboard uses these review statuses: `observation`, `candidate_violation`, `needs_human_review`,
`confirmed_by_reviewer`, and `rejected_by_reviewer`. Reviewer decisions and their change history are
stored in `outputs\review_decisions.sqlite3`. The API maps older report values
(`observation_not_violation` and `confirmed_violation`) to the current names. The offline candidate
analyzer records the relevant frame number, but leaves `evidence_image` empty until an evidence image
has been generated and linked.

Event CSV/JSON downloads include the current review status and decision history. The separate
“Review history” CSV exports each decision change with its previous status, reviewer, note, and time.

## Optional AI features
The upload result offers a video summary, event Q&A, and an individual event explanation. These are
user-triggered and send structured event metadata to the OpenAI Responses API; the source video and
evidence frames are not sent. Conversation history is saved per video job in
`outputs\ai_qa.sqlite3`. The model can explain observations, but cannot confirm a violation; event
references are checked against the supplied event records. Generated summaries and event explanations
are saved as timestamped JSON versions under `outputs\video_jobs`.

1. From `C:\Users\Admin\Desktop\BTECH\RoadSentinel`, copy `.env.example` to `.env`.
2. Set `OPENAI_API_KEY` and `OPENAI_MODEL` in the root `.env` file.
3. Restart the backend. The dashboard enables **Generate summary**, **Event data Q&A**, and **Explain event** when both settings are present.

The trained model defaults to `D:\RoadSentinel-Datasets\training-runs\idd_yolov8n_full\weights\best.pt`.
To use another checkpoint, set `ROAD_SENTINEL_MODEL_PATH` in the backend `.env` file and restart FastAPI.
Upload tracking defaults to confidence `0.35` and Ultralytics `botsort.yaml`. Set
`ROAD_SENTINEL_TRACK_CONFIDENCE` or `ROAD_SENTINEL_TRACKER_CONFIG` to override them. The selected
settings and checkpoint filename are recorded in each job summary JSON.
Candidate event heuristics default to a 3-second minimum track duration, a 5-pixel/second stationary
threshold, and 0.35 minimum detection confidence. Set `ROAD_SENTINEL_EVENT_MIN_DURATION_SECONDS`,
`ROAD_SENTINEL_EVENT_STATIONARY_SPEED_PX_S`, or `ROAD_SENTINEL_EVENT_MIN_CONFIDENCE` to override them.
These values are recorded with the event report; pixel movement remains uncalibrated and cannot establish
real-world speed, road direction, or a violation.

The optional helmet checkpoint defaults to
`D:\RoadSentinel-Datasets\models\helmet_violation_apache_best.pt`; set
`ROAD_SENTINEL_HELMET_MODEL_PATH` to an alternate model or `disabled` to skip this analysis. The model
must expose explicit rider, with-helmet, and without-helmet classes. Helmet inference runs every three
frames by default and requires repeated, sustained no-helmet classifications before producing a review
candidate. Adjust `ROAD_SENTINEL_HELMET_CONFIDENCE`, `ROAD_SENTINEL_HELMET_FRAME_STRIDE`,
`ROAD_SENTINEL_HELMET_MIN_OBSERVATIONS`, `ROAD_SENTINEL_HELMET_MIN_DURATION_SECONDS`, and
`ROAD_SENTINEL_HELMET_MIN_NO_HELMET_RATIO` in the backend `.env` as needed. Scores are detector outputs,
not calibrated probabilities of a violation. The external model file is not bundled; check its upstream
model card and license before redistribution. No independent accuracy benchmark has been performed.

The `.env` file is ignored by Git. Never place the API key in `frontend/.env` or browser code. Each
generated JSON report is saved as a timestamped version under `outputs\video_jobs`, preserving earlier
outputs, and can be downloaded from the dashboard.
