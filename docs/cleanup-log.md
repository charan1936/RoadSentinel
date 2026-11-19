# Repository cleanup log

## 2026-11-05

### Removed

- One-time root patch scripts (`patch_css.py`, `patch_evidence.py`,
  `patch_evidence2.py`, `patch_frontend.py`, `patch_labels.py`,
  `patch_prompt.py`, and `patch_wllm.py`). Repository search found no runtime
  references; their resulting application code remains in the project.
- Zero-byte stray files `dict`, `list[dict]`, and `reviews.sqlite3`. The
  database file was empty; application review data is stored under `outputs/`.
- Stale temporary test directories under `tests/` and Python bytecode caches.
- The prior frontend build output (`frontend/dist/`) after build validation.

### Retained intentionally

- Model checkpoints, dataset directories, sample footage, generated videos,
  analysis outputs, evidence, and databases containing data.
- Historical source/test/style backups and the empty `src/video_detector.py`
  placeholder. They are not runtime dependencies, but are retained because
  this working tree has no committed baseline and their recovery value is
  uncertain.
- `frontend/package-lock.json`, installed `frontend/node_modules/`, and
  `.vscode/settings.json`.

### Ignore rules

Added ignores for common Python/tool caches, TypeScript build metadata, local
SQLite databases, local weights/datasets, backup/temp files, and OS/editor
artifacts. Existing source, tests, documentation, examples, and lockfiles
remain eligible for version control.

### Validation

- Backend: `python -m unittest discover -s tests -v` — 55 tests passed.
- Frontend: `npm run build` — TypeScript check and Vite production build passed.
- Running backend health returned `ok`; the frontend returned HTTP 200 and
  reported API connectivity.
- Persisted completed job `4e2ea2814247` returned a report with all six
  categories and timestamped evidence.
- The valid sample upload/processing workflow and malformed-upload error path
  were exercised in the immediately preceding integration pass; this cleanup
  did not modify runtime application code.
