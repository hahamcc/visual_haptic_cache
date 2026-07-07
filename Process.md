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
