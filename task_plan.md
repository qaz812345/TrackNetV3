## Goal

Refactor the fresh `TrackNetV3` clone into a golf-ball tracking codebase that can train on the current CVAT point annotations and produce reliable per-frame 2D golf-ball detections from behind-player iPhone slow-motion videos.

This is Step 1 of the larger golf trajectory pipeline:

1. Make TrackNetV3 work with golf data.
2. Make the trajectory component work from annotated keypoints.
3. Make the full trajectory component work using TrackNet outputs.


## Current Inputs

- Fresh clone of original `TrackNetV3` repo.
- 2700 CVAT trajectory frames annotated as point labels.
- CVAT XML format is `CVAT for images 1.1`.
- Label: `ball`, type: `points`.
- Attributes:
  - `visibility`: `sharp | blurred | streak | unclear`
  - `usable_for_training`: `yes | no`
- Example image path:
  - `00e53df4-cbd7-4d84-b0ca-e545fa7895a2/frames/trajectory/rel_+000__proposed_rel_-005__frame_003046.jpg`
- Original frame size in the provided annotation snippet: width `1080`, height `1920` portrait.

## Non-goals for this refactor

Do not train InpaintNet first.

Do not build the full 3D trajectory optimizer here.

Do not build backend/API integration here.

Do not hardcode badminton concepts such as `match`, `rally`, `shuttlecock`, or the original dataset layout into new golf code.

## Main refactor strategy

Keep the original TrackNet model architecture usable, but decouple it from the badminton-specific dataset and scripts.

The first working milestone should be:

> CVAT XML + frame folder → golf manifest CSV → TrackNet heatmap dataset → train TrackNet → evaluate on held-out golf shots → predict CSV/JSON on a video or frame sequence.

## Recommended target structure

Create a new package instead of trying to mutate every original script at once:

```text
TrackNetV3/
  tracknet_golf/
    __init__.py
    config.py
    constants.py
    data/
      __init__.py
      cvat_parser.py
      manifest.py
      splits.py
      heatmap.py
      dataset.py
      median.py
    models/
      __init__.py
      tracknet.py
      inpaintnet.py
    training/
      __init__.py
      train_tracknet.py
      losses.py
    inference/
      __init__.py
      predict_video.py
      decode_heatmap.py
      export.py
    evaluation/
      __init__.py
      metrics.py
      evaluate_tracknet.py
    visualization/
      __init__.py
      overlays.py
  scripts/
    convert_cvat_to_manifest.py
    build_golf_splits.py
    train_golf_tracknet.py
    eval_golf_tracknet.py
    predict_golf_video.py
  configs/
    golf_tracknet_baseline.yaml
  tests/
    test_cvat_parser.py
    test_manifest.py
    test_heatmap.py
    test_golf_dataset.py
```

The original files may remain during the transition:

```text
dataset.py
train.py
predict.py
test.py
model.py
utils/
```

But new golf functionality should not depend on the original badminton layout.

## Data design

### Canonical golf manifest

Create one canonical manifest as the bridge between CVAT and training.

Required columns:

```text
sample_id
shot_id
image_id
image_name
image_path
frame_index
rel_frame
proposed_rel_frame
orig_width
orig_height
x_px
y_px
visibility_label
visible
usable_for_training
split
```

Column rules:

- `sample_id`: unique row id, for example `{shot_id}:{frame_index}`.
- `shot_id`: first path component before `/frames/trajectory/`.
- `image_id`: CVAT `<image id="...">` integer.
- `image_name`: raw CVAT image name.
- `image_path`: resolved path to the actual frame file.
- `frame_index`: parse from filename segment like `frame_003046` when possible. Fall back to `image_id` only if parsing fails.
- `rel_frame`: parse from `rel_+000`, `rel_+001`, etc.
- `proposed_rel_frame`: parse from `proposed_rel_-005`, `proposed_rel_+000`, etc.
- `orig_width`, `orig_height`: from CVAT image metadata.
- `x_px`, `y_px`: original-image point coordinate. Use `0,0` if not visible or not annotated.
- `visibility_label`: CVAT `visibility` attribute.
- `visible`: `1` when a usable point exists; otherwise `0`.
- `usable_for_training`: CVAT attribute, normalized to boolean or `yes/no`.
- `split`: assigned by shot-level splitting, never random frame-level splitting.

### Split rule

Split by `shot_id`, not by individual frame.

Reason: adjacent trajectory frames from the same shot are highly correlated. Frame-level splitting leaks almost identical examples between train/val/test.

Suggested initial split:

```text
train: 80% of shots
val:   10% of shots
test:  10% of shots
```

Use deterministic seed, e.g. `seed=13`.

### Frame grouping rule

A TrackNet sample is a temporal sequence from one `shot_id`, sorted by `rel_frame` or `frame_index`.

Initial config:

```yaml
seq_len: 8
sliding_step: 1
padding: false for train/val/test generation
```

For inference on a complete shot/video, allow padding at the end if needed.

## Image-size / orientation decision

The original repo uses:

```python
HEIGHT = 288
WIDTH = 512
```

That is a landscape 16:9 input. Golf frames in the provided annotation snippet are portrait `1080x1920`, so do not keep the original landscape constants blindly.

Start with this baseline because it preserves the 9:16 portrait aspect ratio while keeping the same pixel count as the original repo:

```yaml
input_height: 512
input_width: 288
```

Both dimensions are divisible by 8, so they are compatible with the current TrackNet encoder/decoder pooling/upsampling structure.

Add config support so this can be changed later to a larger portrait input such as:

```yaml
input_height: 672
input_width: 384
```

Only increase resolution after the baseline pipeline works, because memory usage will rise quickly with sequence length and batch size.

## Heatmap target design

The original repo calls `SIGMA = 2.5`, but the implementation generates a binary disk, not a smooth Gaussian:

```python
heatmap[distance_squared <= sigma**2] = 1.0
heatmap[distance_squared > sigma**2] = 0.0
```

For golf, implement heatmap generation as a separate configurable function.

Initial baseline:

```yaml
target_mode: binary_disk
sigma: 2.5
```

Then test:

```yaml
target_mode: binary_disk
sigma: 3.5
```

Optional later experiment:

```yaml
target_mode: gaussian
sigma: 2.0
```

Do not mix this with architecture changes in the same commit. First make one target mode work end-to-end.

## Background handling

The original repo supports background modes:

```text
''
subtract
subtract_concat
concat
```

For golf behind-player tripod videos, background concat is likely useful, but first make the data path work with both no-background and concat modes.

Implementation requirements:

- Compute median per `shot_id` from the frame sequence.
- Store medians under a processed-data directory, not in random source frame folders.
- Dataset should accept `bg_mode=''` and `bg_mode='concat'` at minimum.
- Do not require match-level medians.

Suggested first experiments:

1. `bg_mode=''` to prove baseline.
2. `bg_mode='concat'` after median generation is verified.

## Training scope

Train TrackNet only at first.

Start command shape:

```bash
python scripts/train_golf_tracknet.py \
  --config configs/golf_tracknet_baseline.yaml \
  --manifest data/golf/processed/manifest.csv \
  --save-dir runs/golf_tracknet_baseline
```

Baseline config:

```yaml
seed: 13
input_height: 512
input_width: 288
seq_len: 8
sliding_step: 1
batch_size: 4
epochs: 30
learning_rate: 0.001
optimizer: Adam
bg_mode: ""
target_mode: binary_disk
sigma: 2.5
threshold: 0.5
eval_tolerance_px_input: 4
num_workers: 4
mixed_precision: false
```

Batch size may need adjustment depending on GPU memory.

## Prediction output contract

The golf predictor should output both CSV and JSON.

### CSV

```text
frame_index,rel_frame,x_px,y_px,visibility,confidence,heatmap_score
```

### JSON

```json
{
  "video_id": "string",
  "fps": 240.0,
  "input_size": {"width": 288, "height": 512},
  "original_size": {"width": 1080, "height": 1920},
  "model": {
    "name": "TrackNet",
    "checkpoint": "path-or-id",
    "seq_len": 8,
    "bg_mode": ""
  },
  "detections": [
    {
      "frame_index": 3046,
      "rel_frame": 0,
      "x_px": 534.28,
      "y_px": 1624.03,
      "visibility": 1,
      "confidence": 0.91,
      "heatmap_score": 0.91
    }
  ]
}
```

This is the future bridge to the 3D trajectory optimizer.

## Implementation phases

### Phase 0 — Repo safety and baseline checks

Tasks:

- Run a compile/import check on the untouched repo.
- Record Python and CUDA environment assumptions.
- Add `progress.md`, `findings.md`, and `task_plan.md` to the repo root.

Acceptance criteria:

- Repo state is clean before changes.
- Current files and known hardcoded assumptions are documented in `findings.md`.
- `progress.md` has an initial log entry.

### Phase 1 — Config and package skeleton

Tasks:

- Create `tracknet_golf/` package and `configs/golf_tracknet_baseline.yaml`.
- Move or wrap model creation into `tracknet_golf.models` without changing architecture.
- Keep TrackNet and InpaintNet importable.
- Add a config loader with explicit `input_height`, `input_width`, `seq_len`, `bg_mode`, `sigma`, and paths.

Acceptance criteria:

- `python -m compileall tracknet_golf scripts tests` passes.
- A tiny script can instantiate TrackNet with portrait dimensions and correct input/output channels.
- Original `model.py` behavior is not silently broken.

### Phase 2 — CVAT XML parser

Tasks:

- Implement `tracknet_golf.data.cvat_parser`.
- Parse CVAT image entries, point coordinates, width/height, attributes, shot id, relative frame, proposed relative frame, and actual frame index.
- Handle missing points by creating `visible=0`, `x_px=0`, `y_px=0` rows if such frames exist in CVAT.
- Normalize `usable_for_training`.
- Add unit tests using the provided XML snippet.

Acceptance criteria:

- Parser correctly extracts:
  - `shot_id = 00e53df4-cbd7-4d84-b0ca-e545fa7895a2`
  - `frame_index = 3046` from `frame_003046`
  - `rel_frame = 0` from `rel_+000`
  - `proposed_rel_frame = -5` from `proposed_rel_-005`
  - `x_px = 534.28`, `y_px = 1624.03`
  - `visibility_label = sharp`
  - `usable_for_training = yes`
- Parser returns a DataFrame or list of dataclasses with stable schema.

### Phase 3 — Manifest builder and split generator

Tasks:

- Implement `scripts/convert_cvat_to_manifest.py`.
- Resolve frame paths from a provided frame root.
- Validate that files exist.
- Build one manifest CSV.
- Implement shot-level split assignment.
- Add data quality summary output.

Suggested command:

```bash
python scripts/convert_cvat_to_manifest.py \
  --cvat-xml data/golf/annotations/annotations.xml \
  --frame-root data/golf/raw/frames \
  --out data/golf/processed/manifest.csv \
  --seed 13 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

Acceptance criteria:

- Manifest has exactly one row per CVAT image.
- Rows with `usable_for_training=no` are marked and can be excluded from training.
- Splits are by shot id.
- Summary prints:
  - total frames
  - total shots
  - frames by split
  - shots by split
  - visibility label counts
  - usable/not usable counts
  - missing file count

### Phase 4 — Golf heatmap dataset

Tasks:

- Implement `tracknet_golf.data.dataset.GolfBallTrajectoryDataset`.
- Dataset reads manifest instead of badminton folders.
- Group frames by `shot_id`.
- Create temporal windows of length `seq_len`.
- Resize frames to configured portrait input size.
- Scale coordinates from original size to input size.
- Generate heatmaps with configurable target mode and sigma.
- Return:
  - sequence ids / metadata
  - image tensor
  - heatmap tensor
  - normalized coordinates
  - visibility labels

Acceptance criteria:

- A smoke test can load a 16-row fake manifest and return a correct sample.
- For a visible point, generated heatmap peak is near the scaled coordinate.
- For non-visible point, generated heatmap is all zeros.
- Tensor shapes are correct for:
  - `bg_mode=''`: `(seq_len * 3, H, W)`
  - `bg_mode='concat'`: `((seq_len + 1) * 3, H, W)`

### Phase 5 — Training script for golf TrackNet

Tasks:

- Implement `scripts/train_golf_tracknet.py`.
- Train only TrackNet.
- Use the new dataset and config.
- Save:
  - `TrackNet_best.pt`
  - `TrackNet_cur.pt`
  - config snapshot
  - metrics JSON
  - a few visualizations of input frames + GT heatmaps + predictions
- Support resume training.

Acceptance criteria:

- `--debug` mode trains for 1 epoch on a tiny subset.
- Loss decreases on a tiny overfit subset.
- Checkpoint contains model state and full config.
- Best model chosen by validation F1 or validation accuracy within tolerance.

### Phase 6 — Evaluation for golf detections

Tasks:

- Implement `scripts/eval_golf_tracknet.py`.
- Decode heatmaps to coordinates.
- Compute metrics on val/test split.
- Use input-space tolerance and original-space tolerance in reports.
- Report visibility-specific metrics: sharp, blurred, streak, unclear.
- Save prediction CSV/JSON for each evaluated shot.

Metrics:

- TP: visible GT and prediction within tolerance.
- FN: visible GT but no prediction or outside tolerance.
- FP: no visible GT but prediction exists.
- Mean/median pixel error for visible frames.
- Recall by visibility label.

Acceptance criteria:

- Test/val evaluation produces a JSON summary.
- Evaluation can identify whether failures are mostly streak/blur/unclear cases.
- Predictions can be visually inspected.

### Phase 7 — Golf inference script

Tasks:

- Implement `scripts/predict_golf_video.py`.
- Input: video file or extracted frame directory.
- Output: prediction CSV, prediction JSON, optional overlay video.
- Preserve original frame indexes when possible.
- Support `--impact-frame` or `--start-frame` metadata for downstream trajectory alignment.

Acceptance criteria:

- The script can run on a short golf clip or a folder of trajectory frames.
- It returns per-frame detections in original pixel coordinates.
- JSON output matches the prediction output contract above.

### Phase 8 — Cleanup and naming

Tasks:

- Rename new golf-facing classes/functions away from shuttlecock/badminton language.
- Keep compatibility wrappers only if useful.
- Update README with golf workflow.
- Remove or isolate old corrected badminton test labels from the main path.
- Add clear commands for:
  - preparing data
  - training
  - evaluating
  - predicting

Acceptance criteria:

- A new developer can follow README from CVAT XML to first trained model.
- No new golf scripts require original badminton folder naming.
- Old scripts are either documented as legacy or not part of the golf workflow.

## Claude Code working rules

- Work in small commits / checkpoints.
- After each phase, update `progress.md`.
- When discovering repo-specific facts, update `findings.md`.
- Do not start the next phase until the current acceptance criteria are met or the blocker is documented.
- Prefer tests and smoke scripts over guessing.
- Avoid large architecture rewrites before the dataset/training loop works.
- Do not silently change coordinate conventions. Always state whether coordinates are original-pixel, input-pixel, or normalized.

## Immediate first command sequence

```bash
git status
git checkout -b Phase-1
python -m compileall .
mkdir -p tracknet_golf/{data,models,training,inference,evaluation,visualization} scripts configs tests
find . -maxdepth 2 -type f | sort
```

Then implement Phase 1.
