# Academic submission checklist

Use this list before packaging or submitting the project. Checked items
reflect work verified in the repository as of 2026-11-05; remaining items
require an institutional or project-owner decision.

## Source and reproducibility

- [x] Existing React/TypeScript frontend and FastAPI backend retained.
- [x] Backend dependencies listed in `requirements.txt`.
- [x] Frontend dependencies and lockfile retained (`frontend/package.json`,
  `frontend/package-lock.json`).
- [x] Backend `.env.example` and frontend `.env.example` supplied without
  secrets.
- [x] Windows startup and test commands documented in `README.md`.
- [ ] Confirm the final submission includes the current source files and
  excludes `.env`, runtime database contents, generated uploads, caches, and
  unneeded personal data.
- [ ] Resolve Git tracking before submission: the current repository tracks
  only `.gitignore`; application source, tests, frontend, and documentation
  appear as untracked files. Review and add the intended files without
  including secrets, local databases, model weights, datasets, or generated
  output. Do not bulk-stage the current worktree without review.
- [ ] Record the final source revision/commit or archive checksum if required
  by the institution.

## Dataset and model provenance

- [x] README identifies the external processed IDD/YOLO dataset and YAML
  locations and lists the recorded detector classes.
- [x] README documents the external baseline checkpoint path and states that
  weights are not bundled.
- [x] Validation metrics and incomplete fine-tune are described without
  claiming the latter is superior.
- [ ] Add/attach the dataset source citation, version, license, split counts,
  preprocessing notes, and permission evidence from the dataset owner/source.
- [ ] Add/attach the baseline checkpoint's provenance, license/model card, and
  checksum if redistribution or audit requires it.
- [ ] Keep datasets and original/external checkpoints unchanged; do not
  include assets unless licensing explicitly permits submission.

## Functionality and evaluation

- [x] Backend automated suite: 55 tests passed in the recorded validation.
- [x] Frontend TypeScript check and production build passed.
- [x] Live API health, frontend/API connectivity, completed video processing,
  six-category report, evidence, and malformed-upload error path verified.
- [x] Known analyzer limitations and unavailable/unconfigured states
  documented.
- [ ] Retest on the exact machine, environment, checkpoint, and clips used for
  the final demonstration.
- [ ] If making accuracy claims, conduct an independently annotated,
  representative evaluation and report per-class/category metrics, false
  positives/negatives, and test methodology. Current evidence does not
  provide this.

## Demonstration evidence

- [x] Demo instructions and safe explanation of all six category states
  prepared in `docs/DEMO_GUIDE.md`.
- [ ] Confirm sample-video usage rights and choose the final demo clip.
- [ ] Capture genuine screenshots or a screen recording from the final
  application run if required by the submission.
- [ ] Preserve genuine report/evidence artifacts needed for evaluation in a
  permitted location; do not fabricate or alter detections.
- [ ] Verify the model, output storage capacity, browser playback, and startup
  ports immediately before presentation.

## Academic report and presentation

- [x] Abstract, problem statement, objectives, proposed system, methodology,
  architecture, technology stack, dataset/model summary, results,
  limitations, and future scope drafted in
  `docs/ACADEMIC_PRESENTATION.md`.
- [x] Architecture and Mermaid diagram supplied in `docs/ARCHITECTURE.md`.
- [ ] Add institution-specific front matter, student/supervisor details,
  declarations, acknowledgements, and citation style as required.
- [ ] Verify every reference and add formal citations for datasets, model
  code/checkpoints, frameworks, and any external material used.
- [ ] Distinguish implemented features from optional, experimental,
  unavailable, or calibration-dependent features in slides and oral answers.
- [ ] Proofread the final report and slide deck against the actual running
  application.

## Final safety checks

- [ ] Ensure `.env` and API credentials are not included in archives, slides,
  browser code, screenshots, or source control.
- [ ] Ensure any video/person data shown has appropriate permission and
  handling.
- [ ] State clearly that candidates are for human review and are not legal
  determinations, physical measurements, or collision predictions.
