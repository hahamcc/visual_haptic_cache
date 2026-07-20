# AGENTS.md

## Project Overview

This repository studies visual-to-haptic cache retrieval for robotic tactile prediction.

The project is being rebuilt after previous data and implementation files were lost. The immediate goal is to recover the minimum working loop that existed before the loss:

1. Build aligned visual and tactile records from the VisGel-style dataset.
2. Detect the contact frame from tactile image changes.
3. Localize the GelSight sensor tip and base in RGB frames.
4. Predict a future contact-region heatmap from pre-contact RGB frames and sensor geometry.
5. Retrieve similar historical tactile images/features from a visual-haptic cache.

The long-term goal is low-latency tactile prediction: predict likely contact regions before physical contact, then retrieve cached tactile feedback online to avoid expensive generation whenever possible.

## Research Scope

Core research questions:

- Can future contact regions be predicted from pre-contact video frames?
- Which motion features are useful enough for cache keys: velocity, direction, sensor pose, contact region, or local visual crop?
- Can cached tactile images/features replace a generation model for common contact cases?
- When should the system treat a retrieval as a cache miss and fall back to generation?

Current rebuild priority:

- First restore the 2D minimum baseline.
- Do not introduce SAM, VGGT, 3D reconstruction, large world models, or full contrastive alignment until the baseline is reproducible.
- Prefer simple, inspectable methods first: tactile-difference contact detection, sensor localizer, Gaussian heatmap labels, Tiny U-Net, Top-K proposals, simple KNN retrieval.

## Directory Structure

- `src/`: core Python source code.
- `scripts/`: runnable experiment and utility entrypoints.
- `configs/`: configuration files. Prefer putting paths, thresholds, frame windows, and training parameters here.
- `docs/`: project documentation, paper notes, historical summaries, and previous reasoning.
- `notes/`: active experiment logs, rebuild logs, and Codex working notes.
- `data/`: local datasets and processed manifests. Do not commit large data.
- `outputs/`: local debug images, visualizations, metrics, and experiment outputs. Do not commit generated outputs unless explicitly requested.
- `checkpoints/`: model weights and training checkpoints. Do not commit weights.

## Data Storage Policy

- Keep large raw datasets, copied image folders, videos, historical experiment dumps, and large generated artifacts outside the repository.
- Prefer `/mnt/data/cheng` for large project-owned data and reuse existing raw VisGel data under `/mnt/data/...` when available.
- Keep only small metadata, CSV/JSON indexes, labels, configuration files, scripts, and source code in this repository.
- Heatmaps and debug images may be generated under `data/processed/` or `outputs/` for local inspection, but they should not be committed unless the user explicitly asks.
- When expanding the dataset, scripts should reference large files by absolute path instead of copying them into `data/`.

## Expected Rebuild Modules

Likely modules to rebuild in small steps:

- `src/build_manifest.py`: scan RGB/touch frames and build an aligned manifest.
- `src/detect_contact_frame.py`: detect contact frames from tactile image differences.
- `src/sensor_localizer.py`: train or run the lightweight sensor tip/base localizer.
- `src/build_region_dataset.py`: create pre-contact samples and Gaussian heatmap labels.
- `src/predict_contact_region.py`: train and run the future contact heatmap model.
- `src/build_cache.py`: create visual-haptic cache keys and values from training samples.
- `src/retrieve_cache.py`: retrieve similar cached tactile samples for validation/test samples.
- `src/evaluate.py`: compute region prediction and retrieval metrics.
- `src/visualize.py`: save debug overlays, heatmaps, proposal crops, and retrieval comparisons.

Use these names as guidance, not as a reason to create every file at once. Prefer the smallest module needed for the current task.

## Historical Clues

Known prior work:

- CoTracker was used for motion feature extraction and sensor tracking experiments.
- Pre-contact frames were processed by going backward from the known contact frame.
- A lightweight sensor localizer was trained from a small amount of manual labels.
- The region model used RGB sequence plus sensor geometry and output a future contact heatmap.
- Heatmap Top-K proposals were used for contact region candidates.
- A minimal retrieval loop compared validation samples with training-cache samples.

Known previous metrics from the lost minimum loop:

- median error: about 4.0 px
- PCK@48: about 96.8%
- bbox hit: about 95.5%
- top5 bbox hit: about 99.4%

Known previous output or artifact names:

- `motion_features_precontact.npz`
- `cotracker_tracks_raw_precontact.npy`
- `cotracker_points_precontact.npy`
- `cotracker_precontact_meta.json`

Dataset records may use names like `rec_000xx` or similar episode identifiers.

## Rules for Codex

1. Do not delete files unless the user explicitly asks for deletion.
2. Do not modify `.gitignore` unless the user explicitly asks.
3. Do not commit or push automatically unless the user explicitly asks, except for the daily closeout workflow below.
4. Daily closeout: before ending each work day or long work session, check `git status`, stage only intended source/docs/config files, commit finished work with a clear message, and push the commit to GitHub.
5. During daily closeout, never include large data, generated outputs, model weights, unrelated files, or `docs/` unless the user explicitly asks.
6. Do not put large datasets, videos, images, `.npy`, `.npz`, `.pth`, `.pt`, checkpoints, or generated experiment outputs into Git.
7. Before editing files, explain which files will change and why.
8. Prefer small, incremental changes over large rewrites.
9. Do not revert unrelated user changes.
10. Do not stage unrelated files. In particular, do not stage `docs/` unless the user explicitly asks.
11. Keep configuration values in `configs/` instead of hard-coding paths and thresholds.
12. Keep experiment notes in `notes/experiment_log.md` or `notes/rebuild_log.md` when relevant.
13. After editing, summarize changed files, main logic, how to test, and whether a Git commit is recommended.
14. If a command may require network access, large downloads, or writes outside the workspace, request approval first when required by the environment.

## Coding and Experiment Conventions

- Prefer Python modules with clear command-line entrypoints.
- Use deterministic train/validation/test splits once the split is created.
- Save intermediate metadata as CSV or JSON when practical.
- Save debug overlays for every data-preparation stage before training larger models.
- Make visual checks part of acceptance, not an optional afterthought.
- Keep baseline methods simple until they are reproducible.
- Add concise comments only where the contact-prediction logic is not obvious.

## Current Rebuild Milestones

Phase 1: data and label foundation. Status: basically rebuilt.

- RGB/touch alignment, contact-frame detection, sensor localizer, sensor tracks, and heatmap labels have working rebuild paths.
- The sensor localizer is strong enough for the current loop: test PCK@16 reached 100% in the first rebuilt run.
- Continue to inspect debug overlays whenever new records are added, because bad sensor labels will silently damage later retrieval.

Phase 2 and Phase 3.5: temporal contact-region prediction. Status: usable C2 baseline, not final evaluation.

- The V4 development pool has 4,471 samples: 3,634 train and 837 validation samples across 705 records.
- C2 predicts the future contact box at `probe` values 5, 10, 20, 30, 50, 75, and 100 frames before contact. Four observation frames are input context, not the prediction horizon.
- V4 validation C2 Top-1 Box48 is 96.77%; Top-10 Box48 coverage is 99.64%. For far `probe75/100`, Top-1 is 90.22% and Top-10 coverage is 98.67%.
- Learned Top-K contact rerankers are currently diagnostic only. Their validation gates did not safely improve C2, so deployed contact-box selection remains frozen C2 Top-1 with Top-K retained for uncertainty analysis.
- Split-0 `rec_00950` through `rec_00999` are the sealed final holdout. Do not read their model predictions, inspect their outcomes, or use them for thresholds until the full development recipe is frozen.

Phase 4: local tactile cache retrieval. Status: cache ranking and cache trust are the main bottlenecks.

- Keep the predicted contact box at `48x48`. It provides a specific local visual query while preserving visible localization errors.
- Oracle analysis shows that better tactile cache entries usually exist inside the geometry-filtered train cache, but current rankers do not reliably place them first.
- Never use tactile-cache score alone to choose among C2 Top-K contact boxes: unconstrained tactile selection can move to the wrong physical contact region.
- The useful current cache supervision is soft tactile-embedding listwise ranking. Direct tactile-MAE ranking was rejected because it regressed validation retrieval quality.
- The predicted-box cache ranker is a candidate, not a frozen deployment replacement: it improves MAE/SSIM over the handcrafted key but has mixed IoU behavior, especially on far samples.
- Strict 3-fold OOF cache-confidence data now covers all 3,634 development-train queries. It will train the next cache-trust/cache-miss predictor without query-model leakage.

Next operational step: cache trust and abstention.

- Train a lightweight cache-trust predictor on strict OOF features: ranker best score, ranker margin, handcrafted-key margin, geometry/motion features, probe, and input quality.
- Select its accept/cache-miss threshold on validation only. A cache miss must use a fallback policy rather than returning a forced nearest tactile image.
- Only after the contact model, cache ranker, and trust threshold are fixed may the sealed final holdout be evaluated once.
