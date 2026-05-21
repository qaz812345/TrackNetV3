## Project context

The repo is a fresh clone of original TrackNetV3, which was built for badminton shuttlecock tracking.

The new target is golf-ball tracking from behind-player iPhone slow-motion video. The immediate goal is to retrain TrackNet on 2700 annotated golf trajectory frames and produce per-frame 2D ball detections for the later 3D trajectory optimizer.

The user’s preferred larger implementation order is:

1. Make TrackNetV3 work with golf data.
2. Make the trajectory component work from annotated keypoints.
3. Make the full trajectory component work using TrackNet 

## Uploaded repo inspection summary

Observed files in the provided zip:

```text
README.md
correct_label.py
corrected_test_label/
dataset.py
error_analysis.py
generate_mask_data.py
model.py
predict.py
preprocess.py
requirements.txt
test.py
utils/general.py
utils/metric.py
utils/visualize.py
```

Important consequence: the current repo is script-based and flat. It is not yet organized as a reusable package for a custom golf pipeline.

## Original repo assumptions that must be removed or isolated

### 1. Badminton dataset structure is hardcoded

`dataset.py` has:

```python
data_dir = 'data'
```

The dataset expects folders shaped like:

```text
data/{split}/match{n}/frame/{rally_id}/{frame}.png
data/{split}/match{n}/csv/{rally_id}_ball.csv
```

This does not match the current golf frames, which are organized by shot id and paths like:

```text
00e53df4-cbd7-4d84-b0ca-e545fa7895a2/frames/trajectory/rel_+000__proposed_rel_-005__frame_003046.jpg
```

Decision: create a new manifest-based golf dataset instead of forcing golf data into badminton `match/rally` naming.

### 2. Label CSV format is hardcoded

Original training reads CSV fields:

```text
Frame, Visibility, X, Y
```

The current annotations are CVAT XML point annotations with attributes:

```text
visibility: sharp | blurred | streak | unclear
usable_for_training: yes | no
```

Decision: write a CVAT XML parser and manifest builder. Do not manually convert by hand.

### 3. Original constants are global and landscape-oriented

`utils/general.py` defines:

```python
HEIGHT = 288
WIDTH = 512
SIGMA = 2.5
IMG_FORMAT = 'png'
```

The provided golf annotation snippet has portrait images:

```text
width = 1080
height = 1920
```

Decision: make `input_height` and `input_width` config-driven. Initial golf baseline should use portrait aspect ratio:

```yaml
input_height: 512
input_width: 288
```

This preserves 9:16 orientation and keeps the same approximate input pixel count as the original 288x512 setup.

### 4. Heatmap is binary disk, not Gaussian

The original `_get_heatmap` in `dataset.py` uses distance thresholding:

```python
heatmap[heatmap <= self.sigma**2] = 1.
heatmap[heatmap > self.sigma**2] = 0.
```

So `SIGMA` behaves like disk radius in input pixels, not Gaussian standard deviation.

Decision: extract heatmap generation to `tracknet_golf.data.heatmap` and make target mode explicit:

```yaml
target_mode: binary_disk
sigma: 2.5
```

Later experiments can try larger radius or true Gaussian.

### 5. Preprocess script is not usable for current golf data

`preprocess.py` assumes videos exist under badminton `match/video` folders, extracts frames from `.mp4`, computes match medians, and moves the last rally from each match into validation.

This conflicts with the current dataset because:

- frames are already extracted and annotated;
- data is grouped by shot ids, not matches;
- validation must be split by shot id, not by “last rally”.

Decision: replace preprocessing for golf with:

```text
CVAT XML + frame root -> manifest.csv -> shot-level split -> optional median cache
```

### 6. InpaintNet should be postponed

Original TrackNetV3 includes:

- TrackNet heatmap detector
- InpaintNet trajectory rectifier
- predicted mask generation for rectification

For golf, the next planned component is a physics-aware 3D trajectory optimizer. The badminton trajectory-prior rectifier may not transfer well.

Decision: train/evaluate TrackNet only first. Add InpaintNet later only if clear value remains after physics smoothing.

### 7. Prediction currently writes badminton-style CSV only

`predict.py` writes:

```text
Frame, Visibility, X, Y
```

For the golf pipeline, this is not enough. Need original pixel coordinates, frame indexes, confidence/heatmap score, optional relative frame, and JSON output for downstream trajectory code.

Decision: implement golf prediction output as both CSV and JSON.

## CVAT annotation facts from user snippet

Example image:

```xml
<image id="0" name="00e53df4-cbd7-4d84-b0ca-e545fa7895a2/frames/trajectory/rel_+000__proposed_rel_-005__frame_003046.jpg" width="1080" height="1920">
  <points label="ball" source="manual" occluded="0" points="534.28,1624.03" z_order="0">
    <attribute name="visibility">sharp</attribute>
    <attribute name="usable_for_training">yes</attribute>
  </points>
</image>
```

Expected parse result:

```text
shot_id: 00e53df4-cbd7-4d84-b0ca-e545fa7895a2
image_id: 0
frame_index: 3046
rel_frame: 0
proposed_rel_frame: -5
orig_width: 1080
orig_height: 1920
x_px: 534.28
y_px: 1624.03
visibility_label: sharp
visible: 1
usable_for_training: yes
```

The filename contains two useful frame references:

- `rel_+000`: relative trajectory frame in the exported training frame set.
- `proposed_rel_-005`: relation to the original proposed impact frame.
- `frame_003046`: original/source video frame index.

Use `frame_003046` as canonical `frame_index` for downstream video alignment when available.

## Data leakage risk

There are 2700 annotated trajectory frames. These are sequential frames from shots, likely around 20–30 frames per shot.

Do not split randomly by frame.

Split by `shot_id` so near-duplicate adjacent frames from the same shot do not appear in both train and validation/test.

## Coordinate conventions

Maintain three coordinate spaces explicitly:

1. Original pixel coordinates:
   - From CVAT.
   - Example: `x=534.28`, `y=1624.03` in `1080x1920`.
   - Used by downstream trajectory code.

2. Input pixel coordinates:
   - After resizing to `input_width x input_height`.
   - Used for heatmap generation/evaluation.

3. Normalized coordinates:
   - Usually `[0,1]`.
   - Only use when required by a model or output contract.

Do not mix these silently.

## Initial config decision

Use this as the initial baseline unless a blocker appears:

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

Then run a second experiment with:

```yaml
bg_mode: concat
```

Only after median/background generation is verified.

## Model architecture notes

`model.py` contains:

- `TrackNet(in_dim, out_dim)`: 2D encoder/decoder style network with 3 downsampling stages and skip connections.
- `InpaintNet()`: 1D coordinate rectifier.

TrackNet input channel count depends on sequence length and background mode:

```text
bg_mode=''               -> seq_len * 3 channels
bg_mode='subtract'       -> seq_len channels
bg_mode='subtract_concat'-> seq_len * 4 channels
bg_mode='concat'         -> (seq_len + 1) * 3 channels
```

TrackNet output channels:

```text
out_dim = seq_len
```

Each output channel corresponds to one frame’s heatmap in the sequence.

## Evaluation notes

Original evaluation uses tolerance in input-size pixel space, default `4`.

For portrait baseline `288x512`, input tolerance `4` corresponds to roughly `15` original pixels on a `1080x1920` frame because scale is about `1080/288 = 3.75` and `1920/512 = 3.75`.

Report both:

- input-space pixel error;
- original-space pixel error.

Also report performance by visibility label:

```text
sharp
blurred
streak
unclear
```

This matters because golf balls can be sharp, blurred, or elongated/streaked depending on speed and exposure.

## Windows DLL bug (Phase 4)

Importing `pandas` before `torch` in the same process on Windows causes a fatal DLL initialization error for `torch/lib/c10.dll`. The error also manifests under pytest's assertion-rewriting import mechanism even without explicit pandas import, because `conftest.py` runs before pytest processes test modules.

Fix applied:
1. `dataset.py` imports `torch` before `pandas`.
2. `conftest.py` at repo root imports `torch` at pytest startup so it is already in `sys.modules` when test modules are collected.

Workaround for test invocation: always run `pytest` from repo root; `conftest.py` will handle the rest.

## Environment (Phase 0)

- Python: 3.10.19 via uv venv at `.venv/Scripts/python.exe`
- PyTorch: 2.11.0+cpu (CPU only — no GPU on this machine)
- CUDA: not available
- pip not present in venv Scripts; use `uv pip install <pkg>` to add packages
- pyyaml 6.0.3 installed 2026-05-21

## Phase 1 architecture facts

- `tracknet_golf/config.py` exports `GolfConfig`, `load_config(path)`, `default_config()`.
- `tracknet_golf/config._tracknet_in_dim(bg_mode, seq_len)` computes TrackNet input channels.
- `tracknet_golf/models/tracknet.py` re-imports `TrackNet` from original `model.py` (untouched).
- `build_tracknet(cfg)` is the golf-aware factory function.
- Portrait smoke test: input `(1, 24, 512, 288)`, output `(1, 8, 512, 288)` with `seq_len=8, bg_mode=''`.

## Open questions / later decisions

1. Should `unclear` be excluded from training or included as visible but low-confidence?
   - Initial recommendation: exclude `usable_for_training=no`; include `unclear` only if `usable_for_training=yes`, but report it separately.

2. Should streak annotations use the center of the streak or leading edge?
   - Current annotation convention should be center of visible ball/streak. Keep this consistent.

3. Should input resolution be increased after baseline?
   - Yes, likely. Try `384x672` after the full baseline pipeline works.

4. Should a crop/ROI be used?
   - Later. First train on full portrait frame. ROI/corridor cropping can improve effective resolution but adds another coordinate transform and should not block baseline.

5. Should InpaintNet be trained?
   - Not initially. Use TrackNet detections as input to the custom physics-aware trajectory component first.

## Important implementation warning

The original code resizes with PIL directly to configured width/height. If the wrong orientation is used, portrait video will be squashed into landscape. This would make learning worse and break coordinate intuition.

Always validate a visualization of:

- original frame with CVAT point;
- resized model input with scaled point;
- generated heatmap overlay.
