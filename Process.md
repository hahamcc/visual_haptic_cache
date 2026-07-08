# Process.md

## Current Rebuild Status

The project is being rebuilt after the previous data and implementation were lost.

What remains:

- Project documents and paper notes in `docs/`.
- Basic repository structure.
- `README.md`, `requirements.txt`, and `configs/default.yaml`.

What is currently missing:

- Core implementation under `src/`.
- Runnable experiment scripts under `scripts/`.
- Processed dataset manifests.
- Sensor localizer labels and checkpoints.
- Contact-region model checkpoints.
- Cached visual-haptic retrieval artifacts.
- Previous debug outputs and metric tables.

Immediate objective:

Return to the pre-loss minimum loop: predict future contact regions from RGB sequence plus sensor geometry, produce Top-K contact proposals, and retrieve a similar historical tactile sample from a simple training cache.

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

Once the minimum loop is reproducible:

- Add a lightweight trajectory branch.
- Add mutual constraints between trajectory prediction and contact heatmap prediction.
- Extend the temporal window beyond current frame plus previous 3 frames.
- Optimize online prediction latency.
- Upgrade simple KNN cache retrieval to FAISS if dataset scale requires it.
- Consider stronger visual-tactile alignment only after the baseline is stable.

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
- `outputs/debug/phase2/contact_region/`
- `outputs/debug/phase2/retrieval/`

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
