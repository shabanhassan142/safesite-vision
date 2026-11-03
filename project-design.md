# SafeSite Vision: PPE Compliance Detection System

Full project design (Option 3) for Shaban Hassan. Working name: SafeSite Vision. Change it if you like.

## 1. Project summary

SafeSite Vision takes an image, a recorded video, or a webcam stream from a construction site or factory. It detects workers and their safety gear, tracks each worker over time, and raises a violation when someone is missing a helmet or a safety vest. Every violation is saved with a snapshot and a timestamp. Everything is served through a REST API and a small dashboard.

**Goals**

1. Custom-train a YOLO11 model on PPE data and report real metrics (mAP, precision, recall, per-class results).
2. Add multi-object tracking so one worker gives one violation event, not 300 frames of alerts.
3. Optimize the model for CPU and publish a speed benchmark (PyTorch vs ONNX).
4. Serve it with FastAPI, show it in a dashboard, ship it with Docker.
5. Cover it with proper tests (unit, model, integration, API, performance) and CI.

**Non-goals:** no face recognition, no worker identity. Track IDs are anonymous numbers. Say this in the README. It is a good ethics point.

## 2. Design decisions based on your laptop

Your laptop: Ryzen 5 7430U, 8 GB RAM, integrated Radeon, no NVIDIA GPU.

| Constraint | Decision |
| --- | --- |
| No NVIDIA GPU | Train on Kaggle or Colab free GPU. Run everything else locally on CPU. |
| 8 GB RAM | Deploy YOLO11n (nano). SQLite by default. One API worker. Process every 2nd frame. Close the browser tabs and apps you do not need while testing. |
| CPU inference | Export to ONNX and run with ONNX Runtime. Benchmark at imgsz 640 and 480. |
| Coding agents (Kiro, Cursor) | One small module at a time, with acceptance criteria and tests written first. Details in section 12. |

## 3. Tech stack

| Layer | Choice |
| --- | --- |
| Language | Python&#32; |
| Detection | Ultralytics YOLO11 (PyTorch) |
| Image and video | OpenCV |
| Tracking and drawing | supervision library (ByteTrack, annotators) |
| Export and CPU runtime | ONNX, ONNX Runtime |
| Backend | FastAPI, Uvicorn, Pydantic |
| Database | SQLite via SQLAlchemy (PostgreSQL optional in Docker) |
| Dashboard | Streamlit (calls the API) |
| Testing | pytest, pytest-cov, httpx, ruff |
| Packaging | Docker, docker compose, GitHub Actions |
| Experiment tracking | Plain CSV plus saved plots (or Weights and Biases free tier, optional) |

Flask would also work, but FastAPI gives async upload handling, automatic docs at /docs, and Pydantic validation. You already know it from KnowledgeFlow and CommerceFlow.

## 4. Architecture

```
[Image / Video / Webcam]
          |
          v
   FastAPI backend  <------ Streamlit dashboard
          |
   Inference pipeline
     1. Detector (YOLO11n ONNX)  -> persons + PPE boxes
     2. Tracker (ByteTrack)      -> stable ID per person
     3. Compliance engine        -> PPE-to-person matching, smoothing, rules
     4. Event manager            -> violation open / close, cooldown
          |
          v
   SQLite DB + snapshots folder + annotated video output
```

## 5. Repository structure

```
safesite-vision/
  configs/
    app.yaml            # thresholds, tracker and rule settings
    train.yaml          # model size, imgsz, epochs, batch
  data/                 # NOT committed (gitignored)
    raw/ processed/ samples/
  models/               # best.pt, best.onnx (use Git LFS or release assets)
  notebooks/
    train_kaggle.ipynb  # thin wrapper that calls scripts
  scripts/
    prepare_dataset.py  validate_dataset.py  train.py
    evaluate.py  export_onnx.py  benchmark.py
  src/safesite/
    config.py  schemas.py
    detector.py  tracker.py  compliance.py  events.py
    pipeline.py  video.py  annotate.py
    db/ models.py session.py crud.py
    api/ main.py routes_detect.py routes_jobs.py routes_violations.py routes_stats.py deps.py
  dashboard/
    app.py  pages/
  tests/
    unit/ integration/ api/ model/ performance/ fixtures/
  docker/ Dockerfile.api  Dockerfile.dashboard
  docker-compose.yml
  .github/workflows/ci.yml
  README.md  MODEL_CARD.md  pyproject.toml  requirements.txt
```

## 6. Modules

Each module has a purpose, tasks, a clear output, acceptance criteria, and the tests that prove it works.

### Module 0: Setup

**Purpose:** a clean repo any agent can work in.

**Tasks**

1. Create the repo, virtual environment, pyproject.toml, requirements.txt (pinned versions).
2. Add ruff config, pre-commit, .gitignore (data, models, outputs, .env).
3. Add config loader (src/safesite/config.py) that reads configs/app.yaml into a Pydantic model.
4. Add a logging setup (JSON or plain, one function).

**Acceptance:** `pip install -e .` works, `pytest` runs (even with zero tests), `ruff check` passes.

### Module 1: Data — DONE

**Purpose:** a clean, leak-free dataset in YOLO format.

**Findings:**
- We kept the dataset's own train/valid/test split (2605 / 114 / 82 images).
- Valid and test are ~94% mosaic images with cutout patches; they are not real-world scenes. Evaluation numbers on this split will be optimistic.
- Near-duplicate check (perceptual hash, hamming ≤ 6) found zero cross-split duplicates. Note: hashing cannot detect shared tiles inside mosaics, so tile-level leakage may still exist.

**Datasets to use (check licence and class names yourself before using)**

- Primary: Construction Site Safety dataset (Kaggle, Roboflow export, YOLO format). It has classes like Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest, Person, Mask, NO-Mask, machinery, vehicle.
- Secondary (to add more helmet and vest examples): Hard Hat Workers and Safety Helmet style datasets on Roboflow Universe or Kaggle.

**Final classes (v1, 5 classes)**

| ID | Class |
| --- | --- |
| 0 | person |
| 1 | hardhat |
| 2 | no\_hardhat |
| 3 | safety\_vest |
| 4 | no\_safety\_vest |

Optional v2: mask and no\_mask. Start with 5. Fewer classes means a better model and a finished project.



**Tasks**

1. `prepare_dataset.py`: read each source dataset, remap class names to the 5 final IDs, drop other classes, merge into data/processed.
2. Remove duplicate images (perceptual hash) and images with empty or broken labels.
3. Re-split 70 / 20 / 10 (train / val / test) with a fixed seed. Split by source image group so near-duplicates do not leak across splits.
4. `validate_dataset.py`: check label format, boxes inside 0 to 1, class IDs valid, no image in two splits, report class counts and box size distribution.
5. Write data.yaml and a short EDA notebook or markdown (class balance chart, sample images with boxes).
6. `scripts/extract_frames.py`: reads every video in data/samples, saves every Nth frame (configurable) to data/real_test/images, to be used as the non-augmented held-out test set.

**Outputs:** data/processed with images, labels, data.yaml, and `reports/dataset_report.md`.

**Acceptance:** validator exits with code 0. Each class has at least 500 boxes in train. Zero cross-split duplicates.


### Module 2: Training

**Purpose:** reproducible custom training with comparable experiments.

**Tasks**

1. `train.py` reads configs/train.yaml (model, imgsz, epochs, batch, seed, augmentation) and calls Ultralytics. No hard-coded values.
2. `notebooks/train_kaggle.ipynb`: clone the repo, mount the dataset, run `train.py`, save weights and results as outputs.
3. Run these experiments and log each in `reports/experiments.csv`:

| ID | Model | imgsz | Notes |
| --- | --- | --- | --- |
| E1 | yolo11n | 640 | baseline, 80 epochs, patience 20 |
| E2 | yolo11s | 640 | does a bigger model justify the CPU cost? |
| E3 | yolo11n | 640 | best settings from E1 plus tuned augmentation (mosaic, HSV, close\_mosaic=10) |

**Augmentation note:** the training data is already heavily augmented (≥93% mosaic images, ~61% cutout patches). Use `mosaic=0.0` (or at most `0.3`), keep HSV and `fliplr`, and do not add extra cutout. Set these explicitly in configs/train.yaml so they are not left to Ultralytics defaults.

4. Save best.pt for each run with the config copy beside it.

**Acceptance:** a run can be repeated from the config alone. Results CSV has mAP50, mAP50-95, precision, recall, training time per run.

**Tests:** config loads and validates, training script runs 1 epoch on a 20-image fixture dataset in CI (CPU, under 2 minutes).

### Module 3: Evaluation and error analysis

**Purpose:** prove the model quality honestly, on the untouched test split.

**Tasks**

1. `evaluate.py`: run on the test split, save overall and per-class mAP50, mAP50-95, precision, recall, confusion matrix, PR curves. Report two results:
   - **CSS test split (augmented, optimistic):** the dataset's own test split (mosaic + cutout images).
   - **Real-world test set (non-augmented, held out, never used for training):** frames extracted from data/samples videos and/or a second labeled dataset; labeled separately and never seen during training.
2. Collect the top 30 failure cases per class (false negatives on no\_hardhat and no\_safety\_vest matter most) and save them as a grid image.
3. Tune the confidence threshold per class using the PR curve. Save choices in configs/app.yaml.
4. Write the findings in MODEL\_CARD.md: data sources, classes, metrics, known weaknesses (small workers, night, occlusion), intended use.

**Targets (adjust after the first run):** overall mAP50 of 0.80 or higher, recall of 0.75 or higher on the two no\_ classes.

**Acceptance:** `reports/eval_test.json`, confusion matrix, failure grid, and MODEL\_CARD.md exist.

### Module 4: Export and CPU optimization

**Purpose:** make it fast enough to run on your laptop.

**Tasks**

1. `export_onnx.py`: export best.pt to ONNX (static shape, simplified).
2. `benchmark.py`: on 200 frames from a sample video, measure PyTorch CPU vs ONNX Runtime at imgsz 640 and 480. Record FPS, latency p50 and p95, and peak RAM.
3. Check accuracy loss: run evaluation on the ONNX model and compare mAP to the PyTorch model.
4. Pick the final setup (model, imgsz, frame stride) and save it in configs/app.yaml.

**Acceptance:** `reports/benchmark.md` with a table. ONNX mAP50 within 1 point of PyTorch. Target: 8 FPS or more on your CPU with stride 2 (this is a target, report the real number).

**Stretch:** INT8 quantization with ONNX Runtime and another row in the table.

### Module 5: Inference engine

**Purpose:** one clean detector interface the rest of the app uses.

**Tasks**

1. `detector.py`: class `Detector` with `predict(frame) -> supervision.Detections`. Loads ONNX via Ultralytics or ONNX Runtime. Per-class confidence thresholds and NMS IoU from config.
2. `FakeDetector` in tests: returns scripted detections so most tests need no model file.
3. `video.py`: frame reader generator with stride, resize, and safe release of the capture. Supports file path and webcam index.
4. `annotate.py`: draw boxes, track IDs, green or red status per person, and a violation banner.

**Acceptance:** `Detector` returns correct shapes on a still image. Works on an image, an mp4, and a webcam index. Missing model file gives a clear error.

### Module 6: Tracking and compliance logic (the core of the project)

**Purpose:** turn raw boxes into reliable per-worker violations.

**Step A: Tracking.** Feed only `person` detections to ByteTrack (supervision). Each worker gets a track ID.

**Step B: Match PPE to a person.** For each PPE detection, find the person box that contains its center.

- Helmet classes (hardhat, no\_hardhat): center must be inside the top 40% of the person box.
- Vest classes (safety\_vest, no\_safety\_vest): center must be inside the 20% to 70% height band of the person box.
- If two persons overlap, choose the box with the smaller area that contains the center.

**Step C: Per-frame state.** For every tracked person and each item (helmet, vest), the state is one of OK (positive class seen), MISSING (negative class seen), or UNKNOWN (nothing seen).

**Step D: Temporal smoothing.** Keep the last N=15 processed frames per track. An item is MISSING when at least 60 percent of the frames where it was observed say MISSING, and at least 5 frames were observed. This removes flicker.

**Step E: Event manager.**

1. When an item turns MISSING and no open event exists for (track\_id, item), open a violation event and save a snapshot crop.
2. When the item stays OK for 10 frames, close the event.
3. Cooldown of 10 seconds per (track\_id, item) so the same person cannot spam the log.
4. If a track is lost for 30 frames, close all its open events.

All numbers above live in configs/app.yaml. Tune them on the sample videos.

**Acceptance:** on a scripted sequence, a worker without a helmet for 3 seconds gives exactly 1 violation event. A single-frame false detection gives 0 events.

### Module 7: Storage

**Purpose:** persist jobs and violations.

**SQLite tables**

```
jobs(id, filename, status, progress, total_frames, created_at, finished_at, output_video_path, error)
violations(id, job_id, track_id, violation_type, frame_idx, timestamp_s, confidence, snapshot_path, created_at)
```

violation\_type values: no\_hardhat, no\_safety\_vest.

**Tasks:** SQLAlchemy models, a session dependency, CRUD functions, Alembic migration (optional), and a snapshot folder with safe filenames (UUID based).

**Acceptance:** CRUD tests pass on an in-memory SQLite database.

### Module 8: Backend API (FastAPI)

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | /health | liveness and model-loaded flag |
| GET | /api/v1/model/info | model name, classes, imgsz, thresholds |
| POST | /api/v1/detect/image | upload an image, get detections plus compliance result and optional annotated image |
| POST | /api/v1/jobs/video | upload a video, returns job\_id (runs in a background task) |
| GET | /api/v1/jobs/{id} | status and progress |
| GET | /api/v1/jobs/{id}/video | download the annotated video |
| GET | /api/v1/violations | list with filters: job\_id, type, from, to, limit, offset |
| GET | /api/v1/violations/{id}/snapshot | snapshot image |
| GET | /api/v1/stats/summary | totals, by type, by hour, compliance rate |
| GET | /api/v1/stream | optional MJPEG webcam stream with overlays |

**Rules**

1. Load the model once at startup (lifespan), not per request.
2. Validate file type and size (for example images up to 10 MB, videos up to 200 MB). Reject others with 415 or 413.
3. Run video jobs one at a time (a simple queue) so 8 GB RAM is safe.
4. Pydantic response models for every endpoint. Consistent error format.
5. CORS config and a simple API key header (optional, from env).

**Acceptance:** /docs shows all endpoints. Uploading a sample video gives a finished job, a downloadable annotated video, and violation rows.

### Module 9: Dashboard (Streamlit)

**Pages**

1. **Try it:** upload an image or video, show the annotated result and violation list.
2. **Violations:** filter table with snapshot thumbnails, export CSV.
3. **Analytics:** violations by type, violations per hour, compliance rate, top repeat tracks.
4. **Model:** metrics from the model card, the confusion matrix, and the benchmark table.

The dashboard only talks to the API (httpx). No model code inside Streamlit.

**Acceptance:** the full flow (upload, wait, view violations, see charts) works from the browser.

### Module 10: Testing and CI

See section 10 for the full test plan. This module also sets up the GitHub Actions workflow: ruff, pytest with coverage, upload coverage report, and a Docker build check.

### Module 11: Docker and run guide

1. Dockerfile.api: python:3.11-slim, CPU-only PyTorch install, non-root user, healthcheck on /health.
2. Dockerfile.dashboard: slim image with Streamlit and httpx only.
3. docker-compose.yml: api and dashboard services, volumes for models, data, and snapshots. Optional postgres profile.
4. README commands: `docker compose up --build`, then open the dashboard URL.

**Stretch:** an API image that uses only ONNX Runtime with custom pre and post processing, so the image is much smaller.

**Acceptance:** a clean clone plus `docker compose up` works on a fresh machine with the model file in place.

### Module 12: Docs and portfolio

1. README: problem, demo GIF at the top, architecture diagram, results table, benchmark table, quick start, API examples with curl, project structure, limitations, ethics note.
2. MODEL\_CARD.md from Module 3.
3. 30 to 45 second demo video (annotated output) and 3 screenshots of the dashboard.
4. LinkedIn post: hook, the problem, 3 numbers (mAP50, FPS on CPU, violations detected in the demo), link to GitHub.
5. CV bullets (fill the real numbers after you measure):
   - Fine-tuned YOLO11 on a merged PPE dataset of \[X\] images, reaching \[X\] mAP50 across 5 classes.
   - Built ByteTrack-based compliance engine that cut duplicate alerts by \[X\] percent using temporal smoothing.
   - Optimized for CPU with ONNX Runtime, reaching \[X\] FPS on a Ryzen 5 laptop.
   - Shipped a FastAPI service and Streamlit dashboard with \[X\] tests and Docker deployment.

## 7. Configuration example (configs/app.yaml)

```yaml
model:
  path: models/best.onnx
  imgsz: 640
  iou: 0.5
  conf:
    person: 0.40
    hardhat: 0.35
    no_hardhat: 0.35
    safety_vest: 0.35
    no_safety_vest: 0.35
video:
  frame_stride: 2
  max_upload_mb: 200
tracker:
  lost_buffer: 30
  min_hits: 3
compliance:
  window: 15
  min_observed: 5
  missing_ratio: 0.6
  close_after_ok_frames: 10
  cooldown_seconds: 10
  helmet_zone: [0.0, 0.4]
  vest_zone: [0.2, 0.7]
```

## 8. Testing plan

| Level | What | Examples | Where it runs |
| --- | --- | --- | --- |
| Unit | Pure logic with no model | IoU and center-in-box math, zone matching, smoothing window, event open/close, cooldown, config validation, class mapping | CI and laptop |
| Data | Dataset health | label format, class IDs valid, no cross-split duplicates, min boxes per class | laptop |
| Model | Model sanity | loads, output shape correct, 5 fixture images give the expected classes, ONNX output close to PyTorch output (tolerance), mini test set mAP above a floor | laptop (needs weights) |
| Integration | Pipeline end to end | 3 second fixture video through detector, tracker, compliance, events. Track IDs stay stable. Expected number of events. FakeDetector version runs in CI | CI and laptop |
| API | Endpoints | FastAPI TestClient and httpx: health, image upload, bad file type gives 415, too large gives 413, job lifecycle, violation filters, stats math | CI |
| Performance | Speed and memory | FPS floor on the sample video, peak RAM under 2 GB, p95 latency per frame | laptop |
| End to end | Docker | `docker compose up`, upload a video, poll the job, check violations exist, check snapshot is a valid image | laptop or release check |

**Key edge cases to test**

1. Empty frame with no detections.
2. Two overlapping workers (PPE goes to the right person).
3. Worker leaves and re-enters the frame (new track, no crash).
4. PPE detected with no person box.
5. Corrupt or zero-length video file.
6. Very large image (auto-resized).
7. Same violation continues for 60 seconds (one event, not many).

**Quality gates in CI:** ruff clean, coverage at least 80 percent on src/, all tests pass. Model and performance tests are marked with `@pytest.mark.model` and skipped in CI when the weights are missing.

## 9. Timeline (about 3 weeks with coding agents)

| Week | Work |
| --- | --- |
| 1 | Modules 0 and 1 (setup, data). Module 2 (run E1, E2 on Kaggle). Start Module 3. |
| 2 | Finish 3 and 4 (evaluation, export, benchmark). Modules 5, 6, 7 (engine, tracking, storage) with unit tests. |
| 3 | Modules 8, 9 (API, dashboard). Modules 10, 11 (CI, Docker). Module 12 (docs, demo, posts). |

## 10. Risks and fixes

| Risk | Fix |
| --- | --- |
| Dataset labels are noisy or classes are imbalanced | Validator report, merge a second dataset, class-aware augmentation, report per-class metrics honestly |
| Low recall on no\_hardhat | Lower the class threshold, add data, check the failure grid, smooth over time |
| CPU too slow | imgsz 480, frame stride 3, ONNX Runtime, nano model |
| 8 GB RAM crashes | One worker, one job at a time, stream frames instead of loading the video, SQLite |
| Tracker ID switches | Tune track buffer and min hits, report this as a known limitation |
| Model file too big for GitHub | Use GitHub Releases or Git LFS and a download script |

## 11. Output you should have at the end

1. GitHub repo with clean commits per module.
2. Trained model, model card, evaluation report, benchmark report.
3. Running API, dashboard, Docker setup.
4. Test suite with coverage report and a green CI badge.
5. Demo video, GIF, LinkedIn post, CV bullets with real numbers.

## 12. How to use this with Kiro or Cursor

1. Give the agent this whole document once as context, then work on one module at a time.
2. For each module use this prompt shape: 'Implement Module N exactly as written. Write the tests first. Keep functions small and typed. Do not change other modules. Stop when the acceptance criteria pass and show me the test output.'
3. Commit after each module passes its tests.
4. For Module 6, ask the agent to write the unit tests from the scripted sequences in the acceptance criteria before writing the logic. This is the hardest module, so review it yourself line by line.
5. Do not let the agent invent metrics. All numbers in the README must come from your reports folder.
6. Pin library versions early. Ultralytics and supervision change APIs between versions, so check the agent code against the installed version if something breaks.
