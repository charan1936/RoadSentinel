# RoadSentinel demo guide

## Before the presentation

1. Use the same Windows machine and Python environment on which the model,
   backend packages, and test videos are available.
2. Verify the configured model path and `.env` before the demo. Do not copy
   model weights or datasets into the repository just for the presentation.
3. Use a short local clip from `samples/`, for example `correct video.mp4` or
   `road_video.mp4`. Confirm that you have permission to use/share the clip.
4. Keep a second copy of the clip outside the output directory so it can be
   retried if the backend is restarted.
5. Do not promise that a category will produce an event. The report's
   unavailable, not-configured, and no-events outcomes are valid demo results.
6. If using a previously analyzed job, confirm that its files still exist
   under `outputs/video_jobs/`; saved job history can outlive a removed
   artifact.

## Start the application

### Backend terminal

```bat
cd /d C:\Users\Admin\Desktop\BTECH\RoadSentinel
C:\Projects\RoadSentinel\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8766
```

Wait for Uvicorn startup, then open `http://127.0.0.1:8766/health`. Check
`upload_enabled` and `model_available` are true. If they are false, confirm
`ROAD_SENTINEL_MODEL_PATH` in `.env`, the checkpoint file, and the active
Python environment.

### Frontend terminal

```bat
cd /d C:\Users\Admin\Desktop\BTECH\RoadSentinel\frontend
set "VITE_API_BASE_URL=http://127.0.0.1:8766"
npm run dev -- --port 5174
```

Open `http://127.0.0.1:5174/` and confirm the dashboard reports API
connectivity.

## Upload and processing walkthrough

1. Select **Analyze a new video** / **Upload & analyze**.
2. Choose a clip from `samples/`. The project includes `correct video.mp4`
   and `road_video.mp4`; use a file that has already been checked on the
   presentation machine where possible.
3. Explain that the server validates file extension, size, and video
   readability before queuing it.
4. Show the queued/processing state and progress. Processing time depends on
   clip length, resolution, hardware, and configured model.
5. When complete, show annotated playback, detections, and the event timeline.
6. Select a timestamped candidate. Demonstrate that it opens event/evidence
   details and seeks playback. Explain any image-space measurements as pixels,
   not real-world distance.
7. For wrong-way or restricted-zone parking, calibrate the camera-specific
   region and run the analysis. Each tool shows its own timestamped candidate
   timeline directly below the calibration controls; select an item to open
   evidence and seek playback. A completed analysis with zero candidates is
   shown separately from a calibration that has not been run.
8. Open the **category report** and walk through all six status cards.
9. If an event exists, inspect its evidence and optionally record a reviewer
   decision. Do not use the review action to imply ground-truth validation.

The previously verified sample run for `correct video.mp4` processed
1,022/1,022 frames and produced 3,902 detection rows, 139 unique track IDs, 22
raw candidate-event records, and 22 evidence images. These are processing
counters, not accuracy metrics. In a separate checked `road_video.mp4` run,
the historical report contained 89 unsafe-pedestrian interaction candidates.
Replaying its saved detections with the updated moving-pedestrian filter
produced 60 candidates. Historical reports are not rewritten; results depend
on the exact video, model, configuration, and analyzer version. Do not promise
these counts on a fresh upload.

## Explain the six categories honestly

Use the report status actually shown for the selected job:

| Category | What to explain |
| --- | --- |
| Helmetless riding | Requires a compatible optional helmet-specific checkpoint plus same-frame corroboration from the primary detector's `rider` class. Without it, analysis is unavailable; an empty result is not proof that all riders wore helmets. |
| Triple riding | A spatial/temporal rider-to-motorcycle heuristic generates candidates. A no-events result means its threshold was not met, not that every motorcycle had at most two riders. |
| Restricted-zone parking | Demonstrate drawing/configuring a restricted zone only if time permits. Its candidate timeline appears below the calibration controls. Without calibration, the report correctly says not configured. A stop or stationary detection alone is not proof of illegal parking. |
| Goods-vehicle load risk | The visual cargo-risk detector is unavailable. A truck box is not a weight measurement or evidence of illegal overloading. |
| Unsafe pedestrian interaction | Experimental candidate requiring pedestrian displacement of at least 0.4 median person-box heights, plus closing proximity to a vehicle. The person is shown as the road user and the nearby vehicle separately; it is not physical distance, crossing intent, time-to-collision, or accident prediction. Review the original video. |
| Wrong-way movement | Requires a calibrated road region and expected direction for that camera. Its candidate timeline appears below the calibration controls. Without calibration, it remains not configured; no universal direction is assumed. |

Do not fabricate candidate events, edit results, or describe an empty event
list as a successful negative classification.

## Report walkthrough

Highlight:

- Each category is present, including zero-event and unavailable categories.
- Status indicates whether the analyzer ran, was configured, failed, or
  generated candidates.
- Timestamps, evidence, confidence, and track IDs appear only when recorded.
- Candidates require reviewer inspection; review states are separate from
  model output.
- The report is a video review aid, not a legal finding.

## Troubleshooting

| Symptom | Checks |
| --- | --- |
| Backend does not start | Activate/use the intended Python executable; install `requirements.txt`; verify port `8766` is free. |
| Health says model unavailable | Check `.env`, `ROAD_SENTINEL_MODEL_PATH`, checkpoint existence, and file access. |
| Frontend says API disconnected | Confirm backend health on port `8766`; inspect `frontend/.env` and `VITE_API_BASE_URL`; restart Vite after changing environment. |
| Upload rejected | Use MP4/AVI/MOV/MKV/WEBM, keep below 1 GB, and verify the clip decodes locally. |
| Processing takes too long | Use a shorter local clip; do not start a second inference job. High resolution/CPU fallback can be slow. |
| Browser cannot play annotated video | Wait for conversion to finish; check job state and backend logs; retry with a supported source-video codec. |
| Candidate evidence is missing | Confirm the job completed and evidence artifacts still exist under `outputs/video_jobs/`; do not delete the job outputs during a demo. |
| A category is unavailable/not configured | Explain the missing specialist model or camera calibration. Do not relabel it as a negative result. |

## After the demo

Keep required screenshots/recordings and output evidence only with appropriate
permission and remove or archive generated personal footage according to the
project's data-handling requirements. Runtime job outputs are local and are
not part of the source tree.
