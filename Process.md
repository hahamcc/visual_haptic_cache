# Process.md

## Current Rebuild Status

The project is being rebuilt after the previous data and implementation were lost.

What remains:

- Project documents and paper notes in `docs/`.
- Rebuilt Phase 1 and Phase 2 source code, scripts, configs, and local outputs.
- Manual sensor-localizer labels under `data/makesense/`.
- Processed small metadata under `data/processed/`.
- Current checkpoints and metrics under `checkpoints/` and `outputs/`.

Current recovered capability:

- Phase 1 data and label foundation is working.
- Sensor localizer is usable for current experiments.
- Phase 2 contact-region prediction is no longer the main blocker. On the original 296-sample rebuilt set, the model reached test median error about 6 px and PCK@48/top5 hit@48 about 100%.
- Automatic dataset expansion has a first working version. The first 100-record run produced 530 samples.
- Expanded contact prediction still needs diagnosis: the 100-record expanded run reached test median error about 12 px, PCK@48 about 89.8%, and top5 hit@48 about 100%.

Current main problem:

- Contact-region prediction is reasonably good, especially Top-K proposal coverage.
- Cache retrieval is not good enough. It often retrieves the same object or a visually related object, but not the same local contact position.
- Because tactile images depend strongly on local geometry, retrieving the correct object but the wrong location is still a cache miss for our purpose.

Immediate objective:

Improve the dataset scale and diagnostics first, then improve retrieval so that cache hits match local contact position and not only object-level appearance.

## Phase 1: Data and Label Foundation

Goal: rebuild the data foundation needed for region prediction.

Tasks:

- Build a frame manifest that aligns RGB vision frames with tactile images by record id and frame id.
- Detect contact frames from tactile image changes relative to an initial no-contact tactile frame.
- Recreate or retrain a lightweight sensor localizer for `sensor_tip` and `sensor_base`.
- Derive `sensor_direction_xy` from tip/base coordinates.
- Generate pre-contact training samples using the current frame and a short history window.
- Convert the contact-frame target point into a Gaussian future-contact heatmap.
- Save debug visualizations for sampled records.

Expected artifacts:

- `data/processed/manifest.csv`
- `data/processed/contact_index.csv`
- `data/processed/sensor_tracks.csv`
- `data/processed/region_dataset/`
- `outputs/debug/phase1/`

Acceptance checks:

- Random samples show aligned RGB and touch frames.
- Contact frame visualization matches tactile change.
- Sensor tip/base overlays are visually reasonable.
- Heatmap centers are inside image bounds and near the intended future contact point.
- Train/validation/test split is fixed and reproducible.

## Phase 2: Minimum Prediction and Retrieval Loop

Goal: recover the experiment state that existed before the data loss.

Tasks:

- Train a Tiny U-Net baseline using RGB sequence plus sensor geometry maps.
- Predict a future contact heatmap for validation samples.
- Extract Top-K contact proposals from predicted heatmaps.
- Evaluate contact-region prediction metrics.
- Build a simple cache from training samples using contact region, velocity, direction, and visual crop features.
- Retrieve nearest training-cache samples for validation examples.
- Save visual comparisons of validation crop, retrieved training crop, predicted heatmap, and tactile image.

Expected artifacts:

- `checkpoints/contact_region_baseline/`
- `outputs/metrics/contact_region_baseline.json`
- `outputs/debug/phase2/heatmaps/`
- `outputs/debug/phase2/proposals/`
- `outputs/debug/phase2/retrieval/`

Target metrics from the historical minimum loop:

- median error: about 4.0 px
- PCK@48: about 96.8%
- bbox hit: about 95.5%
- top5 bbox hit: about 99.4%

If these targets are not reached, debug in this order:

1. RGB/touch frame alignment.
2. Contact frame detection quality.
3. Sensor localizer quality.
4. Heatmap label generation.
5. Model input encoding.
6. Top-K proposal extraction.
7. Cache feature design.

## Later Direction

The minimum loop is now reproducible enough to move forward. The next work should be staged carefully.

### Phase 2.5: Dataset Expansion and Diagnosis

Goal: make the training set large enough and trustworthy enough for the next model and retrieval experiments.

Immediate tasks:

- Add time-to-contact bucket metrics, for example near/mid/far groups.
- Compare performance across `probe` values to confirm whether long-horizon prediction is the main source of errors.
- Expand the automatic-label dataset from 100 records to about 200 records as the next controlled run.
- Keep raw RGB/touch data in `/mnt/data/...` or `/mnt/data/cheng`; keep only small labels, CSV/JSON summaries, heatmaps, and debug overlays in this repository.
- Continue using `48x48` contact boxes for visualization and retrieval inspection.

Acceptance checks:

- Automatic-label debug overlays should have green contact boxes centered on plausible true contact regions.
- Expanded training should not only improve train metrics; validation/test PCK@48 and box48 hit must remain stable.
- If top1 worsens while top5 stays strong, inspect Top-K ranking and heatmap sharpness before changing model family.

### Phase 3A: Cache Retrieval Improvement

Goal: make retrieved tactile samples match the query contact location, not just the object.

Current retrieval issue:

- Direct geometry/motion keys are too coarse.
- Handcrafted hybrid crop features help with object appearance but still miss exact local position.
- Query and retrieval can be on the same object type but different part, which makes tactile feedback different.

Next retrieval changes:

- Add explicit local mismatch metrics: distance between query GT contact point and retrieved GT contact point, plus whether the retrieved GT point falls in the query `48x48` box.
- Use two-stage retrieval:
  - Stage 1: filter by contact location, sensor direction, probe/time-to-contact, and motion geometry.
  - Stage 2: re-rank by local `48x48` crop features or learned visual embeddings.
- Treat high-distance retrievals as cache misses instead of forcing a nearest neighbor.
- Keep simple NumPy KNN for now; move to FAISS only when the cache size makes brute force slow.

### Phase 3B: Longer Temporal Prediction

Goal: determine whether more time context improves contact prediction before adding complex interaction models.

Next prediction changes:

- Extend input from the current short history to longer visual/sensor sequences.
- Add explicit trajectory features: tip/base history, velocity, direction, direction stability, and time-to-contact.
- Evaluate by near/mid/far time-to-contact buckets.
- If far-horizon samples improve, keep extending the temporal model; if not, inspect labels and contact-frame detection first.

### Phase 3C: Trajectory-Hotspot Mutual Constraint

Goal: make trajectory and hotspot predictions correct each other.

Planned direction:

- Add a trajectory branch and a hotspot/contact heatmap branch.
- Use trajectory features to constrain likely contact regions.
- Use hotspot/contact features to constrain physically plausible motion endpoints.
- Consider Transformer-style temporal attention after the longer-sequence baseline is stable.
- Do not introduce SAM, VGGT, large models, or full contrastive visual-tactile learning before Phase 2.5 and Phase 3A diagnostics are clear.

## Progress Log Template

Use this format for future updates.

```text
Date:

Completed:
- 

Findings:
- 

Problems:
- 

Next:
- 

Artifacts:
- 
```

## Progress Log

### 2026-07-07 Environment Setup

Completed:

- Installed Miniconda under `/home/cheng/miniconda3`.
- Created the Conda environment `haptic-cache`.
- Installed Python 3.10, NumPy, Pillow, PyYAML, pandas, matplotlib, OpenCV, PyTorch, torchvision, and torchaudio.
- Added `environment.yml` for reproducible environment creation.

Findings:

- The environment uses CPU PyTorch: `torch.cuda.is_available()` is `False`.
- Verified imports for `numpy`, `PIL`, `yaml`, `pandas`, `matplotlib`, `cv2`, `torch`, `torchvision`, and `torchaudio`.
- `python -m py_compile src/*.py` passes inside the Conda environment.

Problems:

- CUDA is not available in the current environment.

Next:

- Use `conda activate haptic-cache` before Phase 2 training work.
- If GPU support is needed later, inspect the system CUDA driver and replace CPU PyTorch with the matching CUDA build.

Artifacts:

- `/home/cheng/miniconda3/envs/haptic-cache`
- `environment.yml`

### 2026-07-07 Sensor Localizer Training

Completed:

- Added a lightweight Tiny U-Net sensor localizer that predicts two heatmaps: `sensor_tip` and `sensor_base`.
- Trained the localizer from `data/processed/sensor_labels.csv` using the Conda environment `haptic-cache`.
- Saved model checkpoints, metrics, predictions, and visual debug overlays.

Findings:

- Training used CPU PyTorch for 80 epochs.
- Split counts are 237 train, 29 validation, and 30 test samples.
- Validation median mean error is about 5.62 px; test median mean error is about 5.14 px.
- Validation PCK@16 and test PCK@16 are both 100%.
- Tip localization is stronger than base localization. Validation tip median is about 2.92 px, while base median is about 8.06 px.

Problems:

- CUDA is still unavailable, so local training is slower than a GPU run.
- The worst validation/test samples mostly come from `sensor_base`, with base errors around 10-15 px.

Next:

- Use this localizer as the Phase 1 sensor geometry baseline.
- If direction quality becomes a bottleneck during contact heatmap training, add a coordinate regression head or upweight the base heatmap loss.

Artifacts:

- `src/train_sensor_localizer.py`
- `scripts/train_sensor_localizer.sh`
- `checkpoints/sensor_localizer/best.pt`
- `outputs/metrics/sensor_localizer_metrics.json`
- `outputs/metrics/sensor_localizer_predictions.csv`
- `outputs/debug/phase1/sensor_localizer_model/`

### 2026-07-07 CUDA Environment Update

Completed:

- Updated `environment.yml` to use CUDA PyTorch with `pytorch-cuda=12.4`.
- Removed CPU-only PyTorch packages from the local `haptic-cache` Conda environment.
- Installed CUDA builds of `pytorch`, `torchvision`, and `torchaudio`.

Findings:

- Conda now has `pytorch 2.4.0 py3.10_cuda12.4_cudnn9.1.0_0`.
- `torch.version.cuda` now reports `12.4`.
- `torch.cuda.is_available()` is still `False` because the system NVIDIA driver is not reachable.
- `nvidia-smi` fails with: cannot communicate with the NVIDIA driver.

Problems:

- The project environment is CUDA-ready, but GPU training still cannot start until the host NVIDIA driver is working in this session.

Next:

- Re-run `nvidia-smi` after the server/driver issue is fixed.
- Once `nvidia-smi` works, verify PyTorch with `torch.cuda.is_available()` before retraining larger models.

Artifacts:

- `environment.yml`

### 2026-07-08 Phase 2 Minimum Loop

Completed:

- Added a Phase 2 contact-region baseline with RGB plus sensor geometry input.
- Added Top-K contact proposal extraction and metrics.
- Added simple NumPy train-cache retrieval.
- Added proposal and retrieval debug visualizations.
- Ran a 2-epoch CPU smoke test in the Codex environment.

Findings:

- The model input has 7 channels: RGB, tip heatmap, base heatmap, direction-x map, and direction-y map.
- The smoke test completed end-to-end and produced checkpoints, metrics, predictions, proposal images, and retrieval images.
- After only 2 CPU epochs, validation median error is about 39.24 px and validation PCK@48 is about 55.17%.
- These smoke-test metrics are not the target result; they only confirm the pipeline works.

Problems:

- Codex still cannot access `/dev/nvidia*`, so formal training should be launched from the user's terminal where `nvidia-smi` works.
- Early 2-epoch predictions still include some edge/corner Top-K false positives.

Next:

- Run `bash scripts/train_contact_region.sh` from a terminal with GPU access.
- Inspect `outputs/debug/phase2/contact_region/` and `outputs/debug/phase2/retrieval/` after the full run.
- If full-run Top-K proposals still drift to image edges, add a simple valid-region mask or tune the heatmap loss.
- Use `48x48` boxes for proposal and retrieval visualization, while keeping PCK@48 for historical comparison.

Artifacts:

- `src/train_contact_region.py`
- `scripts/train_contact_region.sh`
- `checkpoints/contact_region_baseline/best.pt`
- `outputs/metrics/contact_region_baseline.json`
- `outputs/metrics/contact_region_predictions.csv`
- `outputs/metrics/contact_region_retrieval.csv`

### 2026-07-08 Phase 2 Box Visualization Update

Completed:

- Switched proposal and retrieval visualizations from point markers to `48x48` contact-region boxes.
- Changed the simple retrieval crop size from `64x64` to `48x48`.
- Added stricter `box48_hit` and `top5_box48_hit` metrics.
- Added `--eval-only` to regenerate metrics and debug images from an existing checkpoint without retraining.

Findings:

- Re-evaluating the full-run checkpoint gives validation `box48_hit` about 96.55% and validation `top5_box48_hit` 100%.
- Test `box48_hit` and `top5_box48_hit` are both 100%.

Next:

- Use the `48x48` box images when judging whether proposals are specific enough for visual-haptic retrieval.

### 2026-07-08 Cache Retrieval Key Comparison

Completed:

- Added a standalone cache retrieval comparison script.
- Built a `direct` cache key from contact box center, sensor tip/base geometry, motion direction, probe, frame timing, and distance-from-tip features.
- Built a `hybrid` cache key by adding `48x48` contact-region visual features: color statistics, local contrast, edge/texture histogram, and coarse spatial layout.
- Kept the retrieval backend as NumPy brute-force KNN so the result stays simple and inspectable.
- Added CLI overrides for motion and visual weights.

Findings:

- With default weights `motion=1.0` and `visual=1.0`, direct and hybrid retrieval disagree on 33 of 59 validation/test queries.
- The hybrid key reduces median query-GT to retrieved-GT coordinate distance from about 25.61 px to about 16.52 px.
- Hybrid retrieval has lower exact probe match rate than direct retrieval: about 66.10% vs 72.88%.
- Mean direction cosine remains high for both methods, about 0.992, so retrieved examples usually preserve the approach direction.

Problems:

- There is still no true tactile similarity metric, so the current retrieval evaluation is only a proxy plus visual inspection.
- Hybrid sometimes favors visually similar local regions while allowing probe/time-to-contact mismatch.

Next:

- Inspect `outputs/debug/phase2/retrieval_direct/` and `outputs/debug/phase2/retrieval_hybrid/` side by side.
- If hybrid looks better visually, keep the hybrid key and tune motion/probe weighting.
- Add a tactile-image similarity metric before claiming retrieval quality, rather than relying only on coordinate/probe proxies.

Artifacts:

- `src/build_cache_retrieval.py`
- `scripts/build_cache_retrieval.sh`
- `outputs/metrics/contact_region_retrieval_direct.csv`
- `outputs/metrics/contact_region_retrieval_hybrid.csv`
- `outputs/metrics/contact_region_retrieval_compare.csv`
- `outputs/debug/phase2/retrieval_direct/`
- `outputs/debug/phase2/retrieval_hybrid/`

### 2026-07-08 Dataset Expansion Audit

Completed:

- Added a lightweight dataset expansion audit script.
- Scanned split `0` record/frame alignment without copying raw images into the project directory.
- Ran a small contact-detection sample on 20 records from split `0`.
- Added a directory-level overview of available raw VisGel records across all splits.
- Read historical summary files under `/mnt/data/cheng/contact_policy` to recover previous sample-count clues.

Findings:

- Raw VisGel has 10 visible splits, `0` through `9`; each split has 1000 vision record directories and 1000 touch record directories.
- Split `0` alone has 1000 aligned records, and all 1000 are sequence-ready under the current minimum common-frame criterion.
- The current rebuilt dataset uses only 50 labeled records and 296 region samples.
- Split `0` common frames per record: minimum 244, median 364, mean about 318.41, maximum 364.
- In the 20-record contact sample, 15 records produced a contact frame with the current tactile-difference detector.
- Detected records support all current TTC values `[5, 10, 20, 30, 50, 75, 100]`; sequence samples are also mostly available with offsets `[15, 10, 5, 0]`.
- A rough split-0 scale estimate from the small sample is about 5250 TTC samples or about 5200 sequence samples, before applying sensor-localizer and quality filters.
- Historical `/mnt/data/cheng/contact_policy` summaries show previous datasets around 1260-1506 samples, so expanding beyond the current 296 samples is realistic.

Problems:

- Full contact detection over all records is too slow for a short interactive run; it should be run as a longer batch job or split-wise job.
- The 20-record contact sample is only a quick readiness check, not a final quality estimate.
- Expanding labels still depends on robust automatic sensor localization and contact-region quality filtering.

Next:

- Use split-wise expansion rather than copying any large data into this repository.
- First target a controlled expansion to 100-200 records from split `0`.
- Generate small index/label artifacts under `data/processed/`; keep raw RGB/touch data in `/mnt/data`.
- Before Phase 3B/C, convert the expanded data into sequence samples with longer temporal windows.

Artifacts:

- `src/audit_dataset_expansion.py`
- `scripts/audit_dataset_expansion.sh`
- `data/processed/dataset_expansion_audit_records.csv`
- `data/processed/dataset_expansion_contact_sample.csv`
- `outputs/metrics/dataset_expansion_audit.json`

### 2026-07-08 Phase 2.5 Automatic Label Trial

Completed:

- Added an automatic expanded-region dataset builder.
- Reused the trained sensor localizer checkpoint to predict `sensor_tip` and `sensor_base` on raw VisGel frames.
- Used tactile-difference contact detection to find `contact_frame`.
- Used the predicted sensor tip at contact frame as the automatic future contact target.
- Generated `48x48` contact-box heatmaps and Phase 2-compatible sample rows.
- Added quality filters for sensor confidence, tip-base distance, and target-box bounds.
- Ran a 20-record split-0 trial without copying raw RGB/touch files into this repository.

Findings:

- Trial input: split `0`, records `rec_00000` through `rec_00019`.
- Contact detection succeeded for 17 of 20 records.
- The trial produced 119 automatic region samples from 17 records.
- Sample split counts are 98 train, 14 val, and 7 test.
- The sensor localizer produced 374 key-frame track predictions.
- Three records were skipped because contact was not found: `rec_00010`, `rec_00014`, and `rec_00018`.
- No samples failed the current sensor confidence or tip-base-distance filters in this 20-record trial.

Problems:

- The automatic target is currently `predicted sensor tip at contact frame`; this is practical but can be slightly offset from the true tactile contact patch.
- Debug overlays need manual spot checks before scaling to 100-200 records.
- Current contact detection still misses some records and may need threshold or curve-shape tuning before full expansion.

Next:

- Inspect `outputs/debug/phase25/expanded_region_dataset/overlays/`.
- Compare a few automatic targets against the original makesense subset to estimate systematic bias.
- If overlays look acceptable, run the same script with `--record-limit 100` or `--record-limit 200`.
- Train/evaluate the contact-region baseline on the expanded CSV only after the automatic labels pass visual QA.

Artifacts:

- `src/build_expanded_region_dataset.py`
- `scripts/build_expanded_region_dataset.sh`
- `data/processed/expanded_region_dataset/region_samples_auto.csv`
- `data/processed/expanded_region_dataset/sensor_tracks_auto.csv`
- `data/processed/expanded_region_dataset/contact_index_auto.csv`
- `data/processed/expanded_region_dataset/skipped_auto.csv`
- `data/processed/expanded_region_dataset/summary_auto.json`
- `outputs/debug/phase25/expanded_region_dataset/`

### 2026-07-08 Phase 2.6 Expanded Baseline Smoke

Completed:

- Added a separate `contact_region_expanded` config section so expanded training does not overwrite the original 296-sample baseline.
- Added training and retrieval scripts for the expanded dataset.
- Expanded automatic labels from 20 records to 100 records from split `0`.
- Excluded `rec_00007` by default because it was already known as a contact-frame outlier and overlap checks confirmed it causes a large target offset.
- Ran a 2-epoch CPU smoke test for the expanded contact-region baseline.
- Ran expanded direct/hybrid retrieval output generation to verify the evaluation path works.

Findings:

- 100-record automatic labeling selected 100 records and excluded `rec_00007`.
- Contact detection succeeded for 78 records and failed for 21 records.
- 76 records produced usable samples after quality filtering.
- The expanded dataset currently has 530 samples: 426 train, 55 validation, and 49 test.
- Skipped counts: 21 contact-not-found records, 11 low current-tip confidence samples, 3 low target-tip confidence samples, 2 current tip-base distance failures, and 1 excluded record.
- Overlapping automatic labels vs old makesense-derived labels have 244 matched samples.
- After excluding `rec_00007`, overlap target offset is small: mean about 4.25 px, median about 3.95 px, maximum about 9.93 px.
- Overlap contact-frame delta is also small: median 1 frame, range 0-2 frames.
- The 2-epoch smoke run is not a final model, but it verifies training/evaluation/debug/retrieval paths on the expanded CSV.

Smoke metrics:

- Validation median error after 2 epochs: about 21.54 px.
- Validation PCK@48 after 2 epochs: about 74.55%.
- Validation top5 hit@48 after 2 epochs: about 90.91%.

Problems:

- The smoke checkpoint is only a short CPU run; it should not be used for final claims.
- Expanded retrieval numbers from the smoke checkpoint are not meaningful yet because the contact predictor is undertrained.
- Contact detection still misses about one fifth of the first 100 records.

Next:

- Run full expanded training from a GPU terminal: `bash scripts/train_contact_region_expanded.sh`.
- After full training, run `bash scripts/build_cache_retrieval_expanded.sh`.
- Compare expanded model metrics/debug images against the original 296-sample baseline.
- If expanded prediction is stable, scale automatic labeling to 200 records.

Artifacts:

- `scripts/train_contact_region_expanded.sh`
- `scripts/build_cache_retrieval_expanded.sh`
- `checkpoints/contact_region_expanded/`
- `outputs/metrics/contact_region_expanded.json`
- `outputs/metrics/contact_region_expanded_predictions.csv`
- `outputs/metrics/contact_region_expanded_retrieval_direct.csv`
- `outputs/metrics/contact_region_expanded_retrieval_hybrid.csv`
- `outputs/debug/phase26/contact_region_expanded/`

### 2026-07-10 Phase 2 Results and Retrieval Diagnosis

Completed:

- Ran the full contact-region baseline on the original rebuilt 296-sample dataset.
- Added and inspected `48x48` proposal boxes for Top-K contact regions.
- Added direct and hybrid cache retrieval comparisons.
- Built the first automatic-label expanded dataset from 100 records, excluding `rec_00007`.
- Trained the expanded contact-region baseline on 530 automatic-label samples.

Findings:

- Original rebuilt set: test median error is about 6.1 px, PCK@48 is 100%, box48 hit is 100%, and top5 box48 hit is 100%.
- Expanded 100-record set: train median error is 4 px, validation median error is about 8.9 px, and test median error is about 12 px.
- Expanded test PCK@48 is about 89.8%, while top5 hit@48 remains 100% and top5 box48 hit is about 95.9%.
- The contact predictor is useful, but top1 ranking and long-horizon stability need more diagnosis.
- Retrieval remains the main weak point: retrieved samples often match object-level appearance but not exact local contact position.

Problems:

- The current cache key does not represent local tactile identity strongly enough.
- The 100-record expansion is a validation step, not a convincing final dataset scale.
- More samples may improve robustness, but blindly expanding before adding bucket metrics could hide the real failure modes.

Next:

- Add time-to-contact bucket metrics and per-probe summaries.
- Expand the automatic-label dataset to about 200 records for the next controlled run.
- Add retrieval metrics for local mismatch between query contact and retrieved contact.
- Improve retrieval with a two-stage key: geometry/motion/contact filter first, local crop or learned feature re-rank second.
- After these diagnostics, extend the input sequence length and add trajectory features.

Artifacts:

- `outputs/metrics/contact_region_baseline.json`
- `outputs/metrics/contact_region_expanded.json`
- `outputs/metrics/contact_region_retrieval_compare.csv`
- `data/processed/expanded_region_dataset/summary_auto.json`
- `outputs/debug/phase2/contact_region/`
- `outputs/debug/phase2/retrieval/`

### 2026-07-07 Phase 1 Rebuild

Completed:

- Added Phase 1 rebuild code for manifest building, tactile contact detection, makesense sensor label parsing, interpolated sensor tracks, region sample generation, and debug visualization.
- Parsed `data/makesense/images/labels/makesense_labels.csv`.
- Built a labeled-record manifest from `/mnt/data/chi/visgel/seen/images`.
- Generated sensor labels, sensor tracks, contact index, region samples, heatmaps, and Phase 1 debug images.

Findings:

- Makesense labels contain 593 rows, 297 labeled images, and 296 complete `sensor_tip` + `sensor_base` images.
- One image is incomplete: `0_rec_00052_probe050_frame000196.jpg` only has `sensor_tip`.
- The labeled subset covers 50 records and 18,200 aligned RGB/touch frames.
- Contact detection found all 50 labeled records after lowering the tactile-difference threshold to `0.8`.
- Contact detection matches filename-derived contact frames well for most records: median absolute error is 0 frames.
- One clear outlier remains: `rec_00007`, detected contact frame 229 vs filename-derived contact frame 100.
- The region dataset currently has 296 samples: 237 train, 29 val, 30 test.

Problems:

- The current environment has `python3`, `Pillow`, `numpy`, and `PyYAML`, but not `torch`, `cv2`, `pandas`, or `matplotlib`.
- Because `torch` is not available, this phase prepares label/track artifacts but does not train a neural sensor localizer yet.
- `rec_00007` should be manually inspected before it is trusted for model training.

Next:

- User should inspect `outputs/debug/phase1/region_dataset/overlays/` and `outputs/debug/phase1/contact_detection/`.
- After Phase 1 is accepted, start Phase 2: Tiny U-Net or another lightweight contact heatmap baseline.

Artifacts:

- `src/build_manifest.py`
- `src/detect_contact_frame.py`
- `src/sensor_localizer.py`
- `src/build_region_dataset.py`
- `scripts/phase1_rebuild.sh`
- `data/processed/manifest.csv`
- `data/processed/contact_index.csv`
- `data/processed/sensor_labels.csv`
- `data/processed/sensor_tracks.csv`
- `data/processed/region_dataset/region_samples.csv`
- `outputs/debug/phase1/`

### 2026-07-07

Completed:

- Reconstructed project direction from existing `docs/` materials and paper summaries.
- Defined the two-stage rebuild plan: data/label foundation first, then minimum prediction and retrieval loop.
- Added project collaboration instructions and rebuild process documentation.

Findings:

- Current repository has documentation but almost no implementation code.
- The old minimum loop used a sensor localizer, Gaussian heatmap labels, Top-K proposals, and simple cache comparison.
- The main unresolved issue before data loss was precise contact-location discrimination on the same object.

Problems:

- Original data, checkpoints, outputs, and code artifacts are missing.
- `docs/` currently appears as an untracked directory and should not be staged accidentally.

Next:

- Rebuild the data manifest and contact frame detection pipeline.
- Create visual debug outputs before model training.

Artifacts:

- `AGENTS.md`
- `Process.md`
- `README.md`
