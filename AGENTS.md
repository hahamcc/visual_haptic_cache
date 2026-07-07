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

Phase 1: data and label foundation.

- Rebuild RGB/touch manifest.
- Rebuild tactile-difference contact frame detection.
- Rebuild or retrain sensor tip/base localizer.
- Generate pre-contact samples and Gaussian heatmap labels.
- Save visual debug outputs for inspection.

Phase 2: minimum prediction and retrieval loop.

- Train Tiny U-Net or another lightweight baseline for contact heatmap prediction.
- Evaluate median error, PCK@48, bbox hit, top5 bbox hit, and latency.
- Extract Top-K heatmap proposals.
- Build a simple training cache and retrieve nearest historical samples.
- Save retrieval comparison visualizations.

Later phases:

- Add a trajectory branch and contact heatmap branch with lightweight mutual constraints.
- Optimize online latency.
- Consider FAISS, stronger visual features, contrastive alignment, or generation fallback only after the minimum loop is stable.
