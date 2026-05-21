## Current status

Status: Phases 0–4 complete. Next: Phase 5 (training script).

The repo has been inspected at a high level. It is the original TrackNetV3 badminton-oriented codebase with flat scripts and hardcoded dataset assumptions. The next work should start from a safe refactor branch and build a golf-specific data/training path around the existing TrackNet model.

## Active objective

Build the first golf TrackNet training/evaluation loop:

```text
CVAT XML + trajectory frame folders
  -> manifest.csv
  -> shot-level train/val/test split
  -> GolfBallTrajectoryDataset
  -> TrackNet training
  -> golf detection evaluation
  -> prediction CSV/JSON
```

## Completed

- [x] Repo zip inspected.
- [x] Confirmed original repo is badminton/shuttlecock-specific.
- [x] Identified main hardcoded assumptions:
  - `data_dir = 'data'`
  - `HEIGHT = 288`, `WIDTH = 512`, `SIGMA = 2.5`
  - `match/rally/frame/csv` folder structure
  - CSV labels with `Frame, Visibility, X, Y`
  - preprocessing based on videos and badminton matches
- [x] Decided to create a golf-specific manifest-based pipeline instead of forcing golf data into badminton layout.
- [x] Decided to train TrackNet first and postpone InpaintNet.

## Completed

- [x] Phase 0: Compile check passes on original repo.
- [x] Phase 0: Environment recorded — Python 3.10.19 (uv venv), torch 2.11.0+cpu, CUDA unavailable.
- [x] Phase 0: pyyaml installed via `uv pip install pyyaml`.
- [x] Phase 1: `tracknet_golf/` package skeleton created with all subpackages.
- [x] Phase 1: `configs/golf_tracknet_baseline.yaml` created.
- [x] Phase 1: `tracknet_golf/constants.py` with golf-specific defaults.
- [x] Phase 1: `tracknet_golf/config.py` — `GolfConfig`, `load_config()`, `default_config()`, `_tracknet_in_dim()`.
- [x] Phase 1: `tracknet_golf/models/tracknet.py` — re-exports TrackNet + `build_tracknet(cfg)`.
- [x] Phase 1: `tracknet_golf/models/inpaintnet.py` — re-exports InpaintNet.
- [x] Phase 1: `scripts/` and `tests/` stubs created.
- [x] Phase 1: Smoke test passes — portrait `(1, 24, 512, 288)` in → `(1, 8, 512, 288)` out.
- [x] Phase 1: `load_config('configs/golf_tracknet_baseline.yaml')` works.

- [x] Phase 2: `tracknet_golf/data/cvat_parser.py` — 13/13 tests pass.
- [x] Phase 2: `tests/test_cvat_parser.py` — covers all acceptance criteria from task_plan.md.

- [x] Phase 3: `tracknet_golf/data/splits.py` + `tracknet_golf/data/manifest.py` — 11/11 tests pass.
- [x] Phase 3: `scripts/convert_cvat_to_manifest.py` — CLI with summary output.

- [x] Phase 4: `tracknet_golf/data/heatmap.py` — binary_disk + gaussian modes.
- [x] Phase 4: `tracknet_golf/data/dataset.py` — `GolfBallTrajectoryDataset` with all bg_modes.
- [x] Phase 4: 16/16 dataset+heatmap tests pass. 40/40 total.
- [x] Phase 4: `conftest.py` — fixes Windows torch DLL init crash under pytest.

## Not started

- [ ] Phase 5: Implement golf TrackNet training script.
- [ ] Phase 6: Implement golf evaluation script.
- [ ] Phase 7: Implement golf prediction script.
- [ ] Phase 8: Cleanup, README update.

## Implementation log

### 2026-05-21 — Planning handoff

Created planning files for Claude Code / planning-with-files workflow.

Key decision: preserve the original TrackNet model at first, but replace the data/preprocess/training interface with a golf-specific package.

### 2026-05-21 — Phase 0 + Phase 1 complete

Environment: Python 3.10.19 (uv venv), torch 2.11.0+cpu, CUDA unavailable. pyyaml installed via `uv pip install pyyaml`.

Phase 1 smoke test result:
- `build_tracknet(default_config())` → portrait input `(1, 24, 512, 288)` → output `(1, 8, 512, 288)`.
- `load_config('configs/golf_tracknet_baseline.yaml')` parses correctly.
- `python -m compileall tracknet_golf scripts tests` passes with no errors.

### 2026-05-21 — Phase 2 complete

`parse_cvat_xml(path)` returns a DataFrame. All 13 tests pass:
- shot_id, frame_index (from filename `frame_NNNNNN`), rel_frame, proposed_rel_frame parsed correctly.
- x_px/y_px from `points` attribute.
- visibility_label and usable_for_training from `<attribute>` elements.
- visible=0, x=0, y=0 for images with no `<points>` element.
- frame_index falls back to image_id when filename has no `frame_NNNNNN` segment.
- sample_id = `shot_id:frame_index`.

## Next action

Phase 3: Implement `tracknet_golf/data/manifest.py`, `tracknet_golf/data/splits.py`, and `scripts/convert_cvat_to_manifest.py`.

## Current blockers

None yet.

## Risks to watch

### Data leakage

Do not split train/val/test randomly by frame. Split by `shot_id`.

### Orientation bug

Do not use the original landscape `HEIGHT=288`, `WIDTH=512` for portrait golf frames without thinking. Baseline should use:

```yaml
input_height: 512
input_width: 288
```

### Coordinate confusion

Keep original-pixel, input-pixel, and normalized coordinates separate.

### Too much refactor too early

Do not rewrite the whole repo before proving:

```text
one XML snippet -> manifest -> dataset sample -> heatmap -> model forward pass
```

### InpaintNet distraction

Do not train or adapt InpaintNet until TrackNet golf detection is working and evaluated.

## Definition of first successful milestone

A first successful milestone is reached when all of the following are true:

- CVAT XML converts to manifest CSV.
- Manifest rows resolve to real image files.
- Splits are shot-level.
- Dataset returns correct tensors for at least one real shot.
- Heatmap overlay visually matches annotations.
- TrackNet can overfit a tiny subset.
- TrackNet can train on train split and evaluate on val split.
- Evaluation produces metrics and prediction artifacts.
- Prediction output is available in original pixel coordinates.
