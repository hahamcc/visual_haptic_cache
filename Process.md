# Process.md

## Current Rebuild Status

The project is being rebuilt after the previous data and implementation were lost.

## Current Snapshot: 2026-07-20

This section supersedes the older immediate-next-step sections below. Earlier entries remain as historical experiment logs.

### Contact Prediction

- The V4 development pool contains 4,471 samples: 3,634 train and 837 validation samples across 705 records.
- C2 predicts contact regions at `probe` values 5, 10, 20, 30, 50, 75, and 100 frames before contact. The four RGB observations used as context do not mean the model predicts only four frames ahead.
- Validation Top-1 Box48 is 96.77%; Top-10 Box48 coverage is 99.64%.
- Far `probe75/100` validation Top-1 Box48 is 90.22%; Top-10 coverage is 98.67%.
- The current contact-policy baseline is frozen C2 Top-1. Top-K is retained for analysis, but learned contact rerankers have not passed safe validation gates.

### Cache Retrieval

- The contact box is now good enough to isolate the main retrieval issue: the cache often contains a better tactile match, but the ranker does not reliably select it.
- For C2 Top-1 boxes, the frozen cache ranker gives tactile MAE about 0.00955, SSIM about 0.7475, and mask IoU about 0.2260. An offline oracle cache choice at the same predicted box reaches MAE about 0.00707, SSIM about 0.8705, and IoU about 0.3375.
- Tactile similarity alone must not select the contact box: it can improve tactile metrics while choosing a physically wrong local region.
- Soft tactile-embedding listwise supervision is retained. Direct tactile-MAE supervision regressed validation quality and is rejected.
- The latest predicted-box embedding ranker is a candidate only: it improves MAE and SSIM over the handcrafted key, but IoU behavior remains mixed, especially on far samples.

### Cache Trust and Final Evaluation Protocol

- Strict 3-fold OOF cache-confidence outputs now cover all 3,634 train queries. Each fold excludes the held-out query records from cache-ranker training; same-record cache entries are excluded at inference.
- The ranker best score is most useful for MAE/SSIM confidence, while rank margins are more useful for mask-IoU confidence. No single score is safe as a universal cache-miss threshold.
- Next: train a small multi-signal cache-trust/cache-miss predictor on OOF data, then select its threshold on validation only.
- The final holdout is split-0 `rec_00950` through `rec_00999`. It remains sealed until contact model, cache ranker, and trust policy are frozen.

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
- The 100-record expanded baseline reached test median error about 12 px, PCK@48 about 89.8%, and strict Box48 hit about 83.7%.
- The masked real-history C2 model is the current robust temporal baseline: test median error about 12.65 px, PCK@48 about 98.0%, strict Box48 hit about 89.8%, and strict Top-5 Box48 hit about 98.0%.
- On the 14 far samples, C2 reaches median error about 23.32 px, PCK@48 about 92.9%, and strict Top-5 Box48 hit about 92.9%. The remaining weakness is proposal ranking, especially for probe100.

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

### 2026-07-20 V4 Contact-to-Tactile Cache Diagnostics

Completed:

- Built the V4 development pool and strict C2 OOF/validation prediction paths.
- Verified high C2 Box48 and Top-10 coverage at horizons up to 100 frames before contact.
- Ran Phase 4A tactile oracle decomposition, Phase 4B cache-ranker supervision ablations, Phase 4C confidence audit, and Phase 4D strict OOF cache-confidence generation.

Findings:

- Contact-box prediction is no longer the dominant blocker; local cache ranking is.
- A geometry-filtered cache commonly contains a substantially better tactile match than the selected entry.
- Soft tactile-embedding ranking is useful; direct tactile-MAE ranking is not.
- Cache confidence is complementary: absolute ranker score predicts MAE/SSIM better, while score margins predict tactile-mask IoU better.

Problems:

- The predicted-box cache ranker remains only a modest candidate and is not uniformly better on all tactile metrics.
- A single confidence threshold cannot safely identify every bad retrieval.
- No final-holdout claim is valid yet.

Next:

- Train an OOF-supervised multi-signal cache-trust/cache-miss predictor.
- Freeze the resulting threshold on validation, then perform one final-holdout end-to-end evaluation only after the full recipe is fixed.

Artifacts:

- `outputs/metrics/phase4a_topk_tactile_oracle_v4.json`
- `outputs/metrics/phase4b_ablation_predicted_embedding_v4.json`
- `outputs/metrics/phase4c_cache_confidence_audit_v4.json`
- `outputs/cache/phase4d_oof_cache_confidence_v4.csv`

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

### 2026-07-10 Time-to-Contact Bucket Diagnosis

Completed:

- Added `probe` and `ttc_bucket` fields to contact-region prediction CSV output.
- Added per-probe and near/mid/far time-to-contact summaries to contact-region metrics JSON.
- Re-ran eval-only on the original 296-sample baseline and the 100-record expanded baseline.

Findings:

- Original 296-sample test set is stable across time-to-contact buckets:
  - near median error about 6.1 px
  - mid median error about 4.5 px
  - far median error about 5.3 px
  - all buckets have PCK@48 and box48 hit at 100%
- Expanded 100-record test set shows a clear long-horizon failure:
  - near median error about 4.0 px, PCK@48 100%, box48 hit 100%
  - mid median error about 12.6 px, PCK@48 100%, box48 hit 100%
  - far median error about 38.5 px, PCK@48 about 64.3%, box48 hit about 42.9%
- Probe-level expanded test results show the main drop at `probe075` and `probe100`.
- The same hard records are accurate at near probes but drift upward at far probes, which suggests long time-to-contact prediction and Top1 ranking are the main issue.
- Current sensor-localizer confidence does not degrade at far probes, so the failure is not explained by low sensor confidence.
- Since all probes in one record share the same future contact target, the fact that near probes are accurate argues against a global automatic-label failure.

Problems:

- Far-horizon predictions often put Top1 on an upstream location along the approach path.
- Top5 often still contains the correct region, so proposal generation is better than Top1 ranking.
- The current single-frame RGB plus simple geometry input is likely not enough for far time-to-contact samples.

Next:

- Add a time-to-contact-aware model input or loss before expanding too aggressively.
- Start with lightweight changes: include normalized `probe` as an input channel and consider stronger weighting for far samples.
- Then add longer temporal windows and explicit trajectory features.
- Keep automatic labels, but continue visual QA when scaling from 100 to 200 records.

Artifacts:

- `outputs/metrics/contact_region_baseline.json`
- `outputs/metrics/contact_region_expanded.json`
- `outputs/metrics/contact_region_predictions.csv`
- `outputs/metrics/contact_region_expanded_predictions.csv`

### 2026-07-10 Motion-Direction Error Diagnosis and New Ablations

Completed:

- Added actual tip-trajectory motion direction diagnostics using `sensor_tracks.csv` and `sensor_tracks_auto.csv`.
- Projected prediction error into parallel and perpendicular components relative to recent tip velocity.
- Added `motion_dx`, `motion_dy`, `parallel_error`, and `perpendicular_error` fields to prediction CSV outputs.
- Added oracle TTC input-channel support for diagnostic experiment B.
- Added online motion-channel support for deployable experiment C.
- Added two new experiment configs and scripts:
  - `contact_region_expanded_oracle_ttc`
  - `contact_region_expanded_motion`

Findings:

- Expanded test far bucket has a strong negative parallel error:
  - far median parallel error is about `-34.9 px`
  - far parallel negative rate is 100%
  - far median absolute perpendicular error is much smaller, about `12 px`
- `probe075` and `probe100` both have parallel negative rate 100%.
- This supports the hypothesis that far predictions are not randomly drifting left/right; they are usually along the right approach path but not far enough toward the future contact point.
- Original 296-sample baseline does not show the same far-bucket failure: baseline far test parallel negative rate is about 20% and median parallel error is positive.

Problems:

- The 2-epoch oracle TTC and motion-channel runs were only smoke tests and are not meaningful for model comparison.
- Full oracle TTC and motion-channel training should be run from a GPU terminal before drawing conclusions.

Next:

- Run oracle TTC full training:
  - `bash scripts/train_contact_region_expanded_oracle_ttc.sh`
- Run online motion-channel full training:
  - `bash scripts/train_contact_region_expanded_motion.sh`
- Compare far bucket metrics among:
  - `contact_region_expanded`
  - `contact_region_expanded_oracle_ttc`
  - `contact_region_expanded_motion`
- If oracle TTC helps but motion channels do not, the model mainly lacks progress/time-to-contact information.
- If motion channels help, prioritize online trajectory features before Transformer-style temporal models.
- If neither helps and Top5 remains strong, focus on heatmap ranking, coordinate auxiliary loss, or Top5 cache retrieval reranking.

Artifacts:

- `src/train_contact_region.py`
- `configs/default.yaml`
- `scripts/train_contact_region_expanded_oracle_ttc.sh`
- `scripts/train_contact_region_expanded_motion.sh`
- `outputs/metrics/contact_region_expanded_oracle_ttc.json`
- `outputs/metrics/contact_region_expanded_motion.json`

### 2026-07-11 Deployable TTC and Temporal Fusion

Completed:

- Audited current processed data for online controller progress signals.
- Added a standalone 7-bucket TTC estimator using the previous 32 frames of tip/base trajectory.
- Added online-safe trajectory features: absolute and relative tip/base positions, velocity, acceleration, cumulative path length, speed, and direction stability.
- Added TTC bucket accuracy, adjacent-bucket accuracy, TTC MAE, near/mid/far confusion matrix, and remaining-displacement metrics.
- Added experiment C: predicted TTC probability fusion at the Tiny U-Net bottleneck using FiLM.
- Added experiment D: predicted TTC plus a small GRU trajectory branch and remaining-displacement auxiliary loss.
- Added Top-5 physical reranking using heatmap confidence, TTC consistency, lateral deviation, and local `48x48` cache similarity.
- Verified the original expanded baseline remains unchanged after the code extension.

Findings:

- The current CSV data has no planned endpoint, remaining path length, controller ETA, or real action-progress field.
- The start of `sensor_tracks_auto.csv` is selected relative to the detected contact frame, so elapsed time from that artificial start would leak TTC and is deliberately excluded.
- The 100-epoch TTC run selected epoch 7 by validation loss; later epochs overfit strongly.
- Standalone TTC test results are: exact bucket accuracy about 34.7%, adjacent-bucket accuracy about 67.3%, TTC MAE about 17.9 frames, and median remaining-displacement error about 14.8 px.
- For the 14 far test samples, 7 are classified as far and 7 remain classified as mid; trajectory-only TTC is useful but not yet reliable.
- Top-5 reranking with the trained TTC checkpoint does not improve far error: far median stays about 38.5 px and oracle-improvement recovery is 0%.
- The fixed first-pass reranking changes only one test Top1 and makes that sample worse. Reranking weights must be selected on validation only, not tuned against test results.

Next:

- Confirm whether the deployed ROS/robot controller exposes a real online progress or ETA signal. If it does, add it through a separate inference adapter and compare it with trajectory-estimated TTC.
- Use the current TTC checkpoint as a first deployable diagnostic, while treating its exact bucket accuracy as a known limitation.
- Run experiments C and D to determine whether bottleneck fusion can use the probability distribution better than explicit post-hoc reranking.
- Compare A/B/C/D on the same record-level split, especially far median error and near-bucket regression.
- Calculate oracle-improvement recovery; a useful first target is at least 50% recovery while near performance remains stable.
- Revisit Top-5 reranking only after C/D; select all reranking weights on validation and report test once.

Artifacts:

- `src/temporal_progress.py`
- `src/train_ttc_estimator.py`
- `src/rerank_contact_proposals.py`
- `scripts/train_ttc_estimator.sh`
- `scripts/train_contact_region_expanded_predicted_ttc.sh`
- `scripts/train_contact_region_expanded_temporal.sh`
- `scripts/rerank_contact_proposals.sh`
- `outputs/metrics/ttc_estimator.json`
- `outputs/metrics/contact_proposals_reranked.json`

### 2026-07-11 A/B/C/D Full Training Results

Completed:

- Finished 120-epoch experiment C: frozen predicted-TTC probability fusion at the U-Net bottleneck.
- Finished 120-epoch experiment D: jointly trained TTC/trajectory branch with remaining-displacement loss.
- Recomputed headline and far-bucket metrics from the per-sample prediction CSV files.
- Produced a technical experiment artifact at `outputs/reports/phase3_ttc_summary/artifact.json`.

Findings:

- Experiment C is the strongest current direction:
  - overall test median error: `11.31 px`, versus baseline `12.00 px`
  - overall PCK@48: `91.8%`, versus baseline `89.8%`
  - far median error: `15.46 px`, versus baseline `38.54 px` and oracle `16.97 px`
  - far median oracle-improvement recovery: about `107%`
- The C result is not uniformly robust:
  - far PCK@48 improves only from `64.3%` to `71.4%`
  - far 75th-percentile error remains about `51.1 px`
  - far maximum error increases from about `69.1 px` to `80.4 px`
  - failures concentrate in `rec_00092`, `rec_00096`, and `rec_00098`
- Experiment D improves overall median error to `8.00 px` and TTC MAE to `12.10 frames`, but does not solve far prediction:
  - far median error is `38.10 px`
  - overall box48 hit drops to `77.6%`, below baseline `83.7%`
- D shows that better auxiliary TTC/displacement predictions do not automatically produce better hotspot predictions under the current joint objective and bottleneck fusion.

Problems:

- Test coverage is only 7 records and 49 samples; each probe has 7 samples and the far bucket has 14.
- C improves the majority of far samples but leaves a severe failure tail.
- Checkpoint selection still uses validation heatmap MSE, not far median, far PCK@48, or far tail error.
- The portable HTML report could not be packaged because system Node.js is `v12.22.9`, while the official report builder requires newer JavaScript syntax. The canonical report `artifact.json` is preserved.

Next:

- Use C as the current main model and keep the TTC encoder frozen.
- Diagnose the three repeated failure records frame by frame.
- Add validation-time monitoring for far median, far PCK@48, and far 75th-percentile error before changing the loss.
- Expand to at least about 200 records before treating the observed improvement as publication-quality evidence.
- Do not prioritize the current D joint-training design or post-hoc Top-5 reranking until the failure tail is understood.

Artifacts:

- `outputs/metrics/contact_region_expanded_predicted_ttc.json`
- `outputs/metrics/contact_region_expanded_temporal.json`
- `outputs/reports/phase3_ttc_summary/artifact.json`

### 2026-07-11 Failure Audit and E/F Strategy Evaluation

Completed:

- Audited `rec_00092`, `rec_00096`, and `rec_00098` at probe75/probe100 across A, B, C, and C Top-5.
- Added TTC entropy/confidence, contact-frame confidence, multi-scale speed, acceleration, turn angle, trajectory coverage, sensor confidence, and parallel/perpendicular error diagnostics.
- Added A/B/C comparison panels with GT, Top1, and Top-5 `48x48` boxes.
- Added validation-only TTC temperature calibration and confidence-gated strategy E.
- Added validation-only Top-K strategy F using TTC likelihood, lateral deviation, and local cache similarity.
- Extended the automatic track builder so future dataset rebuilds can generate independent 64-frame trajectory history at a 5-frame stride.

Findings:

- The six inspected far samples split into three groups:
  - two `predicted_ttc_resolved` samples from `rec_00096`
  - two low-confidence TTC plus Top-5 ranking failures at probe75 (`rec_00092`, `rec_00098`)
  - two TTC-or-representation failures at probe100 where C Top-5 misses the strict box (`rec_00092`, `rec_00098`)
- Oracle TTC fixes all six inspected samples to within 48 px, so contact labels and basic visual hotspot capacity remain plausible.
- Probe75 failures have normalized TTC entropy around `0.92`, predicted TTC around `30 frames`, and the true region already in C Top-5.
- Probe100 failures have only 4 historical trajectory points spanning 15 frames, despite the model being configured for 32 frames. The remaining history is padded.
- Contact scores are above threshold for all three records, although `rec_00092` has the smallest margin (`0.109`).
- Temperature calibration selected `T=1.25` on validation.
- Strategy E selected confidence threshold `0.15` on validation:
  - overall test median: `8.94 px`
  - overall Box48 hit: `87.8%`, above A (`83.7%`) and C (`85.7%`)
  - near/mid PCK@48 and Box48: `100%`
  - far median: `19.72 px`
  - far P75/P90: `49.05 / 69.14 px`
  - far PCK@48: `71.4%`
  - versus A: 6 wins, 2 losses, 41 ties
- Strategy F selected TTC weight `0`, lateral weight `0.05`, and visual weight `0.1` on validation. This is direct evidence that current candidate-TTC scoring adds no validated value.
- F fails the requested test guardrail: overall Box48 is `81.6%`, below baseline, while far P75 and PCK@48 do not improve.

Problems:

- TTC entropy catches the low-confidence probe75 failures but cannot catch high-confidence probe100 hotspot failures.
- The existing trajectory table does not actually provide 32 historical frames for probe100.
- Validation contains no error above 48 px, while test contains several; threshold/weight selection therefore cannot learn the catastrophic failure mode.
- The current far test set remains only 14 samples from 7 records.

Next:

- Keep E as the safest current inference policy; reject F in its current form.
- Rebuild tracks with at least 64 frames of history before retraining TTC/C/E.
- Expand to about 250-500 effective records to obtain approximately 50-100 far test samples under a 10% record-level test split.
- Preserve A/C/E comparisons and report median, P75, P90, PCK@48, failure rate, max error, per-record results, and win/loss counts.
- Consider adding hotspot-confidence or cache-miss abstention after the longer-history experiment, because TTC confidence alone cannot detect high-confidence representation failures.

Artifacts:

- `src/audit_failure_records.py`
- `src/evaluate_ttc_strategies.py`
- `scripts/audit_failure_records.sh`
- `scripts/evaluate_ttc_strategies.sh`
- `outputs/metrics/failure_record_audit.csv`
- `outputs/metrics/ttc_strategy_evaluation.json`
- `outputs/debug/phase31/failure_records/`

### 2026-07-11 Masked Real-History Rebuild

Completed:

- Added a separate masked-v2 dataset path so previous A/C/E artifacts are not overwritten.
- Rebuilt 100 source records with dense per-frame sensor localization for every `current-31...current` history window.
- Generated 530 region samples and 10,062 real trajectory rows.
- Added per-sample trajectory quality fields: real point count, history span, padding ratio, maximum frame gap, and cumulative displacement.
- Added exact-frame masked trajectory features with `frame_offset` and `is_valid`.
- Added a masked GRU encoder whose hidden state is not updated for invalid/padded frames.
- Trained independent TTC estimators using 8, 16, and 32 real-frame windows.
- Added frozen C-v2 training, structural dual-gate E-v2 evaluation, and Recall@1/5/10/20 evaluation entrypoints.

Findings:

- The previous probe100 history was not a true 32-frame sequence. Four sparse points spanning 15 frames were interpolated, with earlier frames left-filled.
- The rebuilt dataset passes all trajectory coverage checks for every probe:
  - real point count: `32`
  - history span: `31 frames`
  - padding ratio: `0`
  - maximum frame gap: `1`
- The old sparse-history availability pattern was correlated with probe and may have provided an unintended TTC cue.
- Validation-only TTC window comparison:
  - 8 frames: val MAE `24.64`, adjacent accuracy `52.7%`
  - 16 frames: val MAE `24.11`, adjacent accuracy `58.2%`
  - 32 frames: val MAE `24.29`, adjacent accuracy `54.5%`
- The 16-frame estimator is selected because it is best on validation. The lower 32-frame test MAE (`17.60`) is not used for model selection.
- Real-history TTC is harder than the previous sparse/interpolated setup, supporting the concern that the earlier result partially exploited data construction artifacts.
- Existing-model proposal coverage shows:
  - C probe75 strict Box Recall improves from `57.1%` at Top1 to `100%` at Top5.
  - C probe100 remains `71.4%` through Top5 but reaches `100%` at Top10.
  - Oracle TTC reaches `100%` by Top5 for both probe75 and probe100.
- Probe100 therefore still contains usable proposals in ranks 6-10; it is not a Top20 representation failure.

Problems:

- Only 530 samples are available, and all masked GRUs overfit record-level validation quickly.
- The validation difference between 16 and 32 frames is small and should be retested on a larger record split.
- C-v2 has only completed a one-epoch smoke test; its metrics are not meaningful yet.

Next:

- Run the full frozen masked C-v2 training on GPU:
  - `bash scripts/train_contact_region_masked_16.sh`
- After C-v2 finishes, run:
  - `bash scripts/evaluate_dual_gate_masked_16.sh`
  - `bash scripts/evaluate_proposal_recall.sh --section proposal_recall_masked_16`
  - `bash scripts/audit_failure_records.sh --section failure_record_audit_masked_16`
- Accept C-v2/E-v2 only if far P75/P90, probe100 Recall@K, high-confidence severe errors, and overall Box48 improve without near/mid regression.
- Keep quality gating as a structural input contract: the 16-frame model requires 16 real points, 15-frame span, and zero padding. Confidence and temperature remain validation-selected.

Artifacts:

- `data/processed/expanded_region_dataset_masked/`
- `src/temporal_progress.py`
- `src/evaluate_dual_gate.py`
- `src/evaluate_proposal_recall.py`
- `scripts/build_expanded_region_dataset_masked.sh`
- `scripts/train_ttc_estimator_masked.sh`
- `scripts/train_contact_region_masked_16.sh`
- `outputs/metrics/ttc_estimator_masked_8.json`
- `outputs/metrics/ttc_estimator_masked_16.json`
- `outputs/metrics/ttc_estimator_masked_32.json`
- `outputs/metrics/proposal_recall.json`

### 2026-07-11 Masked C2 Full Training and Evaluation

Completed:

- Trained the frozen 16-frame masked C2 model for 120 epochs; validation selected checkpoint epoch 109.
- Evaluated the structural-quality plus TTC-confidence gate E2.
- Recomputed strict Box48 and Euclidean Recall@1/5/10/20 for A, oracle B, and masked C2.
- Audited `rec_00092`, `rec_00096`, and `rec_00098` at probe75 and probe100.

Findings:

- On the same 530-sample expanded split, A versus C2 test performance is:
  - A: median `12.00 px`, PCK@48 `89.8%`, strict Box48 `83.7%`.
  - C2: median `12.65 px`, PCK@48 `98.0%`, strict Box48 `89.8%`, strict Top-5 Box48 `98.0%`.
- Far performance improves substantially from A to C2:
  - median error: `38.54 -> 23.32 px`
  - PCK@48: `64.3% -> 92.9%`
  - strict Box48: `42.9% -> 64.3%`
- Compared with the old confidence-gated E result, C2 slightly worsens far median (`19.72 -> 23.32 px`) but greatly reduces tail risk:
  - P75: `49.05 -> 32.25 px`
  - P90: `69.14 -> 34.18 px`
  - error above 48 px: `28.6% -> 7.1%`
- Near and mid retain `100%` PCK@48 and strict Box48. Mid median is `17.89 px`, so its coordinate precision still has room to improve despite passing the box criterion.
- C2 still predicts too little distance along the correct motion path on far samples: median parallel error is `-19.89 px`, while median absolute perpendicular error is `12.40 px`.
- Probe75 is mostly a Top-1 ranking problem: strict Box48 Recall is `85.7%` at Top1 and `100%` at Top5.
- Probe100 needs a wider candidate set: strict Box48 Recall is `42.9%` at Top1, `85.7%` at Top5, and `100%` at Top10.
- `rec_00098` probe100 remains the only error above 48 px (`80.10 px`). Its correct strict contact region is already in Top-5, so this is a ranking failure rather than a missing-proposal failure.
- E2 selected confidence threshold `0.0` on validation. Every rebuilt input also passes the fixed history-quality contract, so E2 selects C2 for every sample and produces identical test metrics.
- TTC softmax confidence does not separate C2 wins from losses. Keep the history-quality gate as an input validity check, but do not treat TTC confidence gating as a performance improvement.

Problems:

- The far test set still contains only 14 samples, so per-probe percentages move by about 14.3 points per sample.
- C2 improves robustness but does not reach oracle B; probe100 Top-1 localization and candidate ranking remain weak.
- The single catastrophic `rec_00098` probe100 case has turning/acceleration behavior and low direction stability. Its predicted TTC is far too short even though the true region exists among proposals.
- Confidence-only fallback is ineffective on this validation split and should not be tuned on test failures.

Next:

- Keep C2 as the current deployable temporal baseline and keep E2 only as a structural malformed-history safeguard.
- Use Top-10 proposals for far/probe100 experiments; Top-5 is insufficient under the strict `48x48` contact box.
- Train a small frozen trajectory-endpoint model or learned Top-K ranker that scores endpoint probability, lateral consistency, local visual similarity, and cache similarity. Do not let predicted TTC control the full heatmap.
- Add an explicit cache-miss/abstention output when proposal and retrieval confidence are insufficient.
- Evaluate the new ranker first on validation, then once on test, reporting far median/P75/P90, strict Box48, Recall@K, and per-record failures.
- After the ranker interface is stable, expand to roughly 250-500 effective records so the far test set reaches at least 50-100 samples.

Artifacts:

- `checkpoints/contact_region_masked_16/best.pt`
- `outputs/metrics/contact_region_masked_16.json`
- `outputs/metrics/contact_region_masked_16_predictions.csv`
- `outputs/metrics/dual_gate_masked_16.json`
- `outputs/metrics/proposal_recall_masked_16.json`
- `outputs/metrics/failure_record_audit_masked_16.csv`

### 2026-07-11 Top-10 Learned Proposal Ranker

Completed:

- Added a frozen-C2 Top-10 candidate extraction and learned ranking pipeline.
- Added candidate features from heatmap rank/score, candidate geometry, tip-relative motion coordinates, predicted endpoint, TTC, trajectory quality, and the local `48x48` visual crop.
- Added record-level 3-fold out-of-fold C2 training so the ranker never trains on contact-model predictions from records seen by that contact model.
- Generated OOF proposals for all 426 original training queries across 61 records. The original test records are excluded from OOF training.
- Added far-only Box48 candidate supervision and validation-only rerank-margin selection. Near/mid always preserve C2 Top1.
- Added a safe fallback: when no validation-supported rerank margin improves the result, inference keeps the original C2 Top1.

Findings:

- The ordinary ranker setup was invalid because full C2 has `100%` train Top1 Box48 and therefore supplies no correction examples.
- Record-level OOF fixes that leakage and produces 121 far queries, but only 7 are true hard cases where Top1 misses and Top-10 contains a correct strict Box48 candidate.
- OOF far Top-10 strict Box48 coverage is `100%`, confirming that candidate generation remains adequate.
- A local-visual ranker can learn some useful corrections on OOF data: one diagnostic run improved OOF far Box48 from `94.2%` to `97.5%`.
- That diagnostic ranker reduced `rec_00098` probe100 test error from `80.10 px` to `52.00 px`, but did not move it into the correct strict Box48 region.
- Removing heatmap/rank features was rejected because validation Box48 fell from `87.3%` to `80.0%` and test failure rate increased.
- The final validation-gated model makes no automatic test changes. It preserves C2 metrics exactly:
  - overall median `12.65 px`, PCK@48 `98.0%`, strict Box48 `89.8%`
  - far median `23.32 px`, P90 `34.18 px`, PCK@48 `92.9%`, strict Box48 `64.3%`
- This is a safe negative result: the ranking infrastructure works, but the present data does not support a ranker that generalizes beyond C2.

Problems:

- Seven OOF hard cases are too few for a 119-feature local ranking problem.
- Existing records are dominated by easy near/mid queries; the useful ranking supervision is concentrated in probe75/100.
- The remaining severe case involves turning/acceleration, which is weakly represented in the current 61 OOF training records.
- Because the validation-supported gate rejects all corrections, the proposal-ranker phase has not passed research acceptance and cache retrieval should not yet assume a corrected Top1 box.

Next:

- Move dataset expansion ahead of final ranker acceptance. Expand to roughly 250-500 effective records while keeping raw data under `/mnt/data/cheng` or the existing `/mnt/data/chi` source.
- Prioritize far, turning, acceleration, deceleration, and visually ambiguous same-object contact locations instead of only adding easy near samples.
- Require at least 50-100 OOF hard cases and at least 50 far test samples before retraining the learned ranker.
- Preserve the current OOF split, Top-10 interface, validation-only margin gate, and C2 fallback during expansion.
- Resume the two-stage tactile cache only after the ranker improves validation far Box48/P90 without reducing PCK@48 or increasing severe failures.

Artifacts:

- `src/build_oof_contact_proposals.py`
- `src/train_proposal_ranker.py`
- `scripts/build_oof_contact_proposals.sh`
- `scripts/train_proposal_ranker_masked_16.sh`
- `scripts/run_proposal_ranker_phase.sh`
- `outputs/metrics/proposal_ranker_oof_masked_16.json`
- `outputs/metrics/proposal_ranker_masked_16.json`
- `outputs/metrics/proposal_ranker_masked_16_predictions.csv`
- `outputs/debug/phase34/proposal_ranker_masked_16/`

### 2026-07-11 Phase 35 Data Protocol and Oracle Tactile Cache

Completed:

- Defined the next experiment around effective hard examples instead of total query count.
- Added a fixed record-level split manifest to the expanded-dataset builder. The manifest is written before sensor localization, contact detection, or heatmap generation.
- Added the Phase 35 250-record-v2 configuration with `199` train, `25` validation, and `25` final-holdout records from seed `20260711`.
- Final-holdout records are constrained to `rec_00100+`, preventing records previously used by C2 development from entering the final holdout.
- Kept large Phase 35 heatmaps and debug images under `/mnt/data/cheng/haptic-cache/phase35_250_v2/`; repository-local outputs remain CSV/JSON summaries and code only.
- Added an oracle-contact-box two-stage retrieval experiment: GT contact box -> geometry filter -> local visual rerank -> tactile comparison.
- Added tactile evaluation baselines: difference-map MAE, global SSIM, deformation-mask IoU, contact-area difference, deformation-centroid distance, and fixed difference-map embedding distance.

Findings:

- The current 100-record test set is now a diagnostic set because `rec_00092/96/98` has influenced repeated design decisions. It must not be reported as the final unbiased result.
- The fixed Phase 35 holdout is not inspected for prediction outcomes while data, ranking features, and thresholds are being developed.
- Oracle-box retrieval on the current validation split (`55` queries, `426` train-cache entries) is not yet a satisfactory tactile cache upper bound:
  - deformation-mask IoU: `0.126` versus random `0.092`
  - tactile difference-map MAE: `0.01160` versus random `0.01147`
  - tactile SSIM: `0.630` versus random `0.632`
- Therefore correct local visual contact boxes alone do not make the current handcrafted geometry-plus-visual cache key retrieve tactile states reliably. The cache representation and cache-miss confidence require further work in parallel with ranking.
- The current cache-miss threshold is also not semantically reliable: items marked as miss have better tactile metrics on this development split than items marked as hit.

Problems:

- Ranker training needs `rank-hard` examples, not more easy near/mid examples.
- The current data cannot establish an unbiased final result.
- The oracle tactile evaluation uses a fixed handcrafted difference-map embedding, not a learned tactile encoder; it is a useful baseline but not the final cache representation.

Next:

- Finish the fixed 250-record build, then inspect only train/validation for sensor/contact quality and count far, `easy`, `rank-hard`, and `proposal-miss` samples.
- Do not inspect final-holdout model predictions. If the training pool has fewer than 50 `rank-hard` OOF examples, expand to 500 records before retraining the ranker.
- Retrain the ranker with hard-query upweighting, positive-versus-current-Top1 pairwise ranking loss, and easy-case stability loss.
- Select rerank gate only on validation, then evaluate the untouched final holdout once.
- Improve oracle tactile retrieval independently with stronger local visual/tactile features and calibrated cache-miss logic before using predicted boxes for end-to-end cache claims.

Artifacts:

- `data/processed/expanded_region_dataset_phase35_250/record_splits_fixed.csv`
- `src/build_expanded_region_dataset.py`
- `scripts/build_expanded_region_dataset_phase35_250.sh`
- `src/evaluate_oracle_tactile_retrieval.py`
- `scripts/evaluate_oracle_box_tactile_retrieval.sh`
- `outputs/metrics/oracle_box_tactile_retrieval.json`
- `outputs/metrics/oracle_box_tactile_retrieval.csv`

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
