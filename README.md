# TrackNetV3: Enhancing ShuttleCock Tracking with Augmentations and Trajectory Rectification
We present TrackNetV3, a model composed of two core modules: trajectory prediction and rectification. The trajectory prediction module leverages an estimated background as auxiliary data to locate the shuttlecock in spite of the fluctuating visual interferences. This module also incorporates mixup data augmentation to formulate complex
scenarios to strengthen the network’s robustness. Given that a shuttlecock can occasionally be obstructed, we create repair masks by analyzing the predicted trajectory, subsequently rectifying the path via inpainting.
[[paper](https://dl.acm.org/doi/10.1145/3595916.3626370)]

<div align="center">
    <a href="./">
        <img src="./figure/NetArch.png" width="50%"/>
    </a>
</div>

## Performance 

* Performance on the test split of [Shuttlecock Trajectory Dataset](https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw).

<div align="center">
    <table>
    <thead>
        <tr>
        <th>Model</th> <th>Accuracy</th> <th>Precision</th> <th>Recall</th> <th>F1</th> <th>FPS</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>YOLOv7</td> <td>57.82%</td> <td>78.53%</td> <td>59.96%</td> <td>68.00%</td> <td><b>34.77</b></td>
        </tr>
        <tr>
        <td>TrackNetV2</td> <td>94.98%</td> <td><b>99.64%</b></td> <td>94.56%</td> <td>97.03%</td> <td>27.70</td>
        </tr>
        <tr>
        <td>TrackNetV3</td> <td><b>97.51%</b></td> <td>97.79%</td> <td><b>99.33%</b></td> <td><b>98.56%</b></td> <td>25.11</td>
        </tr>
    </tbody>
    </table>
    </br>
    <a href="./">
        <img src="./figure/Comparison.png" width="80%"/>
    </a>
</div>

## Installation
* Develop Environment
    ```
    Ubuntu 16.04.7 LTS
    Python 3.8.7
    torch 1.10.0
    ```
* Clone this reposity.
    ```
    git clone https://github.com/qaz812345/TrackNetV3.git
    ```

* Install the requirements.
    ```
    pip install -r requirements.txt
    ```

## Inference
* Download the [checkpoints](https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view?usp=sharing)
* Unzip the file and place the parameter files to ```ckpts```
    ```
    unzip TrackNetV3_ckpts.zip
    ```
* Predict the label csv from the video
    ```
    python predict.py --video_file test.mp4 --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir prediction
    ```
* Predict the label csv from the video, and output a video with predicted trajectory
    ```
    python predict.py --video_file test.mp4 --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir prediction --output_video
    ```
* For large video
    * Enable the ```--large_video``` flag to use an IterableDataset instead of the normal Dataset, which prevents memory errors. Note that this will decrease the inference speed.
    * Use ```--max_sample_num``` to set the number of samples for background estimation.
    * Use ```--video_range``` to specify the start and end seconds of the video for background estimation.
    ```
    python predict.py --video_file test.mp4 --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir prediction --large_video --video_range 324,330
    ```

## Training
### 1. Prepare Dataset
* Download [Shuttlecock Trajectory Dataset](https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw)
* Adjust file structure:
    1. Merge the `Professional` and `Amateur` match directories into a single `train` directory.
    2. Rename the `Amateur` match directories to start from `match24` through `match26`.
    3. Rename the `Test` directory to `test`.
* Dataset file structure:
```
  data
    ├─ train
    |   ├── match1/
    |   │   ├── csv/
    |   │   │   ├── 1_01_00_ball.csv
    |   │   │   ├── 1_02_00_ball.csv
    |   │   │   ├── …
    |   │   │   └── *_**_**_ball.csv
    |   │   ├── frame/
    |   │   │   ├── 1_01_00/
    |   │   │   │   ├── 0.png
    |   │   │   │   ├── 1.png
    |   │   │   │   ├── …
    |   │   │   │   └── *.png
    |   │   │   ├── 1_02_00/
    |   │   │   │   ├── 0.png
    |   │   │   │   ├── 1.png
    |   │   │   │   ├── …
    |   │   │   │   └── *.png
    |   │   │   ├── …
    |   │   │   └── *_**_**/
    |   │   │
    |   │   └── video/
    |   │       ├── 1_01_00.mp4
    |   │       ├── 1_02_00.mp4
    |   │       ├── …
    |   │       └── *_**_**.mp4
    |   ├── match2/
    |   │ ⋮
    |   └── match26/
    ├─ val
    |   ├── match1/
    |   ├── match2/
    |   │ ⋮
    |   └── match26/
    └─ test
        ├── match1/
        ├── match2/
        └── match3/
```
* Attributes in each csv files: `Frame, Visibility, X, Y`
* Data preprocessing
    ```
    python preprocess.py
    ```
* The `frame` directories and the `val` directory will be generated after preprocessing.
* Check the estimated background images in `<data_dir>/median`
    * If available, the dataset will use the median image of the match; otherwise, it will use the median image of the rally.
    * For example, you can exclude `train/match16/median.npz` due to camera angle discrepancies; therefore, the dataset will resort to the median image of the rally within match 16.
* Set the data root directory to `data_dir` in `dataset.py`.
    * `dataset.py` will generate the image mapping for each sample and cache the result in `.npy` files.
    * If you modify any related functions in `dataset.py`, please ensure you delete these cached files.
### 2. Train Tracking Module
* Train the tracking module from scratch
    ```
    python train.py --model_name TrackNet --seq_len 8 --epochs 30 --batch_size 10 --bg_mode concat --alpha 0.5 --save_dir exp --verbose
    ```

* Resume training (start from the last epoch to the specified epoch)
    ```
    python train.py --model_name TrackNet --epochs 30 --save_dir exp --resume_training --verbose
    ```

### 3. Generate Predicted Trajectories and Inpainting Masks
* Generate predicted trajectories and inpainting masks for training rectification module
    * Noted that the coordinate range corresponds to the input spatial dimensions, not the size of the original image.
    ```
    python generate_mask_data.py --tracknet_file ckpts/TrackNet_best.pt --batch_size 16
    ```

### 4. Train Rectification Module
* Train the rectification module from scratch.
    ```
    python train.py --model_name InpaintNet --seq_len 16 --epoch 300 --batch_size 32 --lr_scheduler StepLR --mask_ratio 0.3 --save_dir exp --verbose
    ```

* Resume training (start from the last epoch to the specified epoch)
    ```
    python train.py --model_name InpaintNet --epochs 30 --save_dir exp --resume_training
    ```

## Evaluation
* Evaluate TrackNetV3 on test set
    ```
    python generate_mask_data.py --tracknet_file ckpts/TrackNet_best.pt --split_list test
    python test.py --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir eval
    ```

* Evaluate the tracking module on test set
    ```
    python test.py --tracknet_file ckpts/TrackNet_best.pt --save_dir eval
    ```

* Generate video with ground truth label and predicted result
    ```
    python test.py --tracknet_file ckpts/TrackNet_best.pt --video_file data/test/match1/video/1_05_02.mp4 
    ```

## Error Analysis Interface
* Evaluate TrackNetV3 on test set and save the detail results for error analysis
    ```
    python test.py --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir eval --output_pred
    ```

* Add json path of evaluation results to the file list in `error_analysis.py`
    ```
    30  # Evaluation result file list
    31  if split == 'train':
    32      eval_file_list = [
    33          {'label': label_name, 'value': json_path},
     ⋮                              ⋮
            ]
        elif split == 'val':
            eval_file_list = [
                {'label': label_name, 'value': json_path},
                                    ⋮
            ]
        elif split == 'test':
            eval_file_list = [
                {'label': label_name, 'value': json_path},
                                    ⋮
            ]
        else:
            raise ValueError(f'Invalid split: {split}')                                  
    ```

* Run Dash application
    ```
    python error_analysis.py --split test --host 127.0.0.1
    ```
<div align="center">
    <a href="./">
        <img src="./figure/ErrorAnalysisUI.png" width="70%"/>
    </a>
</div>

## Reference
* TrackNetV2: https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2
* Shuttlecock Trajectory Dataset: https://hackmd.io/@TUIK/rJkRW54cU
* Labeling Tool: https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2?tab=readme-ov-file#label


# TrackNetV3_TableTennis Inference Pipeline 重構說明（Phase 1–3）
##### 目前環境的修改
```
cd TrackNetV3_TableTennis/
conda create -n tracknetv3_311 python=3.11 -y
conda activate tracknetv3_311
sed -i 's/==/>=/g' requirements.txt
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
pip install pycocotools
pip install psutil
pip install prefetch_generator
pip install av

python predict.py --video_file 048/C0050.mp4 --tracknet_file exp/TrackNet_best.pt --inpaintnet_file exp/InpaintNet_best.pt --save_dir 048 --eval_mode weight --output_video --large_video --max_sample_num 60 >>demo.txt
```
本文件記錄對 [`wasn-lab/TrackNetV3_TableTennis`](https://github.com/wasn-lab/TrackNetV3_TableTennis) 推論流程三個階段的修改，所有「Before / After」皆為實際 diff，從 repo 原始版本對照當前修改版本。

修改範圍：

- `dataset.py` — `Video_IterableDataset` 的 `__iter__`、`__gen_median__`、`__process__`，以及新增 helper
- `utils/general.py` — `draw_traj`、`write_pred_video`，新增 `FFmpegWriter` 與 `_COLOR_MAP`

`Shuttlecock_Trajectory_Dataset` 在此三階段中**未動**。

---

## Phase 1：Sliding window 冗餘計算消除 + `write_pred_video` 改用 FFmpeg

### 1.1 `Video_IterableDataset.__iter__` 重寫

**Before（原版）**

每輸出一個 sample 就呼叫一次 `self.__process__(np.array(frame_list)[..., ::-1])`，把整個 `seq_len` 幀的 list 重新處理（resize + bg subtract + normalize）一次：

```python
def __iter__(self):
    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success = True
    start_f_id, end_f_id = 0, 0
    frame_list = []
    while success:
        # Sample frames
        while len(frame_list) < self.seq_len:
            success, frame = self.cap.read()
            if not success:
                break
            frame_list.append(frame)
            end_f_id += 1

        # Form a sequence
        data_idx = [(0, i) for i in range(start_f_id, end_f_id)]
        if len(data_idx) < self.seq_len:
            data_idx.extend([(0, end_f_id-1)] * (self.seq_len - len(data_idx)))
            frame_list.extend([frame_list[-1]] * (self.seq_len - len(frame_list)))
        data_idx = np.array(data_idx)
        frames = self.__process__(np.array(frame_list)[..., ::-1])  # ← 整段重新處理
        yield data_idx, frames

        # Update the sliding window
        frame_list = frame_list[self.sliding_step:]
        start_f_id = start_f_id + self.sliding_step

    self.cap.release()
```

問題：在 `sliding_step=1` 時，第 i+1 個 sample 與第 i 個 sample 共用了 `seq_len-1` 幀，但這些幀仍然會被重新 resize、subtract、normalize 一次。前處理量級為 `O(N × seq_len)`。

**After（新版）**

改成 deque-based 串流：每幀只執行一次 `_preprocess_one_frame`，之後存進 deque buffer。新增 sample 時只讀取 `sliding_step` 個新幀。

```python
def __iter__(self):
    """串流讀取：每幀只 preprocess 一次，用 deque 做滑動視窗"""
    from collections import deque
    cap = cv2.VideoCapture(self.video_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    buf = deque(maxlen=self.seq_len)  # 存已處理好的 (C,H,W) float32
    next_read_f_id = 0

    # 先填滿 buffer
    while len(buf) < self.seq_len:
        success, frame = cap.read()
        if not success:
            break
        buf.append(self._preprocess_one_frame(frame))
        next_read_f_id += 1
    if len(buf) == 0:
        cap.release()
        return

    # 若影片比 seq_len 短，做 padding
    while len(buf) < self.seq_len:
        buf.append(buf[-1].copy())

    # 主迴圈:每次輸出一個 sample，再 slide
    while True:
        end_f_id = next_read_f_id
        seq_start = end_f_id - self.seq_len
        data_idx = np.array([(0, seq_start + i) for i in range(self.seq_len)])
        frames = self._assemble(buf)
        yield data_idx, frames

        # 滑動 sliding_step：丟掉前 sliding_step 幀，讀入 sliding_step 新幀
        produced_new = 0
        for _ in range(self.sliding_step):
            success, frame = cap.read()
            if not success:
                break
            buf.append(self._preprocess_one_frame(frame))  # deque maxlen 自動丟最舊
            next_read_f_id += 1
            produced_new += 1
        if produced_new < self.sliding_step:
            break
    cap.release()
```

新增兩個 helper：

- `_preprocess_one_frame(frame_bgr)` — 處理單幀，依 `bg_mode` 回傳 `(C, H, W) float32 in [0,1]`，C 為 1 / 3 / 4。
- `_assemble(buf)` — 把 deque 拼成最終輸出 shape，`bg_mode='concat'` 時前置 median。

複雜度從 `O(N × seq_len)` 降為 `O(N)`。

### 1.2 `__process__` 向量化

**Before（原版）**

PIL 逐幀處理，每幀都做一次 `Image.fromarray` → 操作 → `np.array`：

```python
def __process__(self, imgs):
    if self.bg_mode:
        median_img = self.median
    frames = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)
    for i in range(self.seq_len):
        img = Image.fromarray(imgs[i])
        if self.bg_mode == 'subtract':
            img = Image.fromarray(np.sum(np.absolute(img - median_img), 2).astype('uint8'))
            img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
            img = img.reshape(1, self.HEIGHT, self.WIDTH)
        elif self.bg_mode == 'subtract_concat':
            diff_img = Image.fromarray(np.sum(np.absolute(img - median_img), 2).astype('uint8'))
            diff_img = np.array(diff_img.resize(size=(self.WIDTH, self.HEIGHT)))
            diff_img = diff_img.reshape(1, self.HEIGHT, self.WIDTH)
            img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
            img = np.moveaxis(img, -1, 0)
            img = np.concatenate((img, diff_img), axis=0)
        else:
            img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
            img = np.moveaxis(img, -1, 0)
        frames = np.concatenate((frames, img), axis=0)

    if self.bg_mode == 'concat':
        frames = np.concatenate((median_img, frames), axis=0)
    frames /= 255.
    return frames
```

問題：

1. 逐幀 `Image.fromarray` ↔ `np.array` 來回，每幀兩次完整 memcpy。
2. PIL `resize` 比 `cv2.resize` 慢約 3–5 倍。
3. `np.concatenate` 在迴圈中累加，是 `O(seq_len²)` 的暫存配置。

**After（新版）**

整段改寫為 numpy batched ops：

```python
def __process__(self, imgs):
    """Process the frame sequence — vectorised rewrite (concat mode optimised)."""
    # 1. 批次 resize：cv2.resize 比 PIL.resize 快約 3-5x
    resized = np.empty((self.seq_len, self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
    for i in range(self.seq_len):
        resized[i] = cv2.resize(imgs[i], (self.WIDTH, self.HEIGHT),
                                interpolation=cv2.INTER_LINEAR)

    if not self.bg_mode:
        # (seq_len, H, W, 3) → (seq_len, 3, H, W) → (seq_len*3, H, W)
        frames = np.ascontiguousarray(
            resized.transpose(0, 3, 1, 2).reshape(-1, self.HEIGHT, self.WIDTH))

    elif self.bg_mode == 'subtract':
        median = self._get_median_hwc()                          # (H,W,3) uint8
        diff = np.abs(resized.astype(np.int16) - median.astype(np.int16))
        diff = diff.sum(axis=3).astype(np.uint8)                 # (seq_len, H, W)
        frames = diff[:, np.newaxis, :, :]                       # (seq_len, 1, H, W)
        frames = frames.reshape(-1, self.HEIGHT, self.WIDTH).astype(np.float32)

    elif self.bg_mode == 'subtract_concat':
        median = self._get_median_hwc()
        diff = np.abs(resized.astype(np.int16) - median.astype(np.int16))
        diff = diff.sum(axis=3).astype(np.uint8)
        rgb  = resized.transpose(0, 3, 1, 2)                     # (seq_len, 3, H, W)
        diff = diff[:, np.newaxis, :, :]                         # (seq_len, 1, H, W)
        frames = np.concatenate([rgb, diff], axis=1)             # (seq_len, 4, H, W)
        frames = frames.reshape(-1, self.HEIGHT, self.WIDTH).astype(np.float32)

    else:  # 'concat'
        rgb = resized.transpose(0, 3, 1, 2)
        frames = rgb.reshape(-1, self.HEIGHT, self.WIDTH)
        median = self._get_median_chw()                          # (3, H, W) float32
        frames = np.concatenate([median, frames], axis=0)

    return frames.astype(np.float32) / 255.
```

並新增兩個 median 格式 helper，避免每幀重複轉軸：

```python
def _get_median_hwc(self):
    """median を (H, W, 3) uint8 で返す（形式を統一）"""
    m = self.median
    if m.shape[0] == 3:                  # (3, H, W) → (H, W, 3)
        m = m.transpose(1, 2, 0)
    return m.astype(np.uint8)

def _get_median_chw(self):
    """median を (3, H, W) float32 で返す"""
    m = self.median
    if m.ndim == 3 and m.shape[2] == 3:  # (H, W, 3) → (3, H, W)
        m = m.transpose(2, 0, 1)
    return m.astype(np.float32)
```

`__init__` 進一步在類別建構時就預先計算並快取：

```python
if self.bg_mode in ('subtract', 'subtract_concat'):
    self._median_hwc_i16 = self._get_median_hwc().astype(np.int16)
if self.bg_mode == 'concat':
    self._median_chw_f32 = self._get_median_chw().astype(np.float32) / 255.0
```

之後 `_preprocess_one_frame` 與 `_assemble` 直接拿這兩個快取用，整個推論期間不再重複轉型。

注意：`__process__` 在新版實際上**只在訓練/測試 path 用**；推論 path 改走 `_preprocess_one_frame` + `_assemble`。`__process__` 留著是為了向後相容（仍能被舊版 `__iter__` 風格的程式呼叫）。

### 1.3 `write_pred_video` 改用 FFmpeg subprocess

**Before（原版）**

`cv2.VideoWriter` + `mp4v` fourcc，純 CPU 編碼：

```python
def write_pred_video(video_file, pred_dict, save_file, traj_len=8, label_df=None):
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))                      # ← int() 會丟失小數
    w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if label_df is not None:
        f_i, x, y, vis = label_df['Frame'], label_df['X'], label_df['Y'], label_df['Visibility']
    x_pred, y_pred, vis_pred = pred_dict['X'], pred_dict['Y'], pred_dict['Visibility']

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_file, fourcc, fps, (w, h))

    if not out.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for mp4. ...")

    pred_queue = deque()
    if label_df is not None:
        gt_queue = deque()

    i = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        if len(pred_queue) >= traj_len:
            pred_queue.pop()
        if label_df is not None and len(gt_queue) >= traj_len:
            gt_queue.pop()

        if label_df is not None:
            gt_queue.appendleft([x[i], y[i]]) if vis[i] and i < len(label_df) else gt_queue.appendleft(None)
        pred_queue.appendleft([x_pred[i], y_pred[i]]) if vis_pred[i] else pred_queue.appendleft(None)

        if label_df is not None:
            frame = draw_traj(frame, gt_queue, color='red')
        frame = draw_traj(frame, pred_queue, color='yellow')

        if len(pred_queue) > 0 and pred_queue[0] is not None:
            cx, cy = pred_queue[0]
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"Frame: {i}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        out.write(frame)
        i += 1

    out.release()
    cap.release()
```

問題：

1. `cv2.VideoWriter` 走 CPU 編碼，4K 60fps 寫出比推論還慢。
2. `fps = int(...)` 強制取整，29.97 fps 影片會寫成 29 fps，造成時序漂移。
3. 沒有對 `len(x_pred)` 做邊界檢查；若預測比影片短會 IndexError。

**After（新版）**

新增 `FFmpegWriter` 類別，介面相容 `cv2.VideoWriter`，底層用 subprocess pipe：

```python
class FFmpegWriter:
    """
    用 ffmpeg subprocess 做硬體/軟體編碼，介面相容 cv2.VideoWriter。
    吃 BGR uint8 ndarray，輸出 mp4。

    codec:
        'h264_nvenc' : NVIDIA GPU 硬編（最快，需要 NVENC 支援）
        'libx264'    : CPU 軟編（相容性最佳）
    """
    def __init__(self, save_file, width, height, fps,
                 codec='h264_nvenc', preset=None, cq=23):
        common_in = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{int(width)}x{int(height)}',
            '-r', f'{fps}',
            '-i', '-', '-an',
        ]
        common_out = ['-pix_fmt', 'yuv420p', '-movflags', '+faststart', save_file]

        if codec == 'h264_nvenc':
            preset = preset or 'p4'   # p1(最快) ~ p7(品質最好)
            enc = ['-c:v', 'h264_nvenc', '-preset', preset, '-cq', str(cq)]
        elif codec == 'libx264':
            preset = preset or 'veryfast'
            enc = ['-c:v', 'libx264', '-preset', preset, '-crf', str(cq)]
        else:
            raise ValueError(f'Unsupported codec: {codec}')

        self.proc = subprocess.Popen(common_in + enc + common_out,
                                     stdin=subprocess.PIPE, ...)

    def write(self, frame_bgr):
        if not frame_bgr.flags['C_CONTIGUOUS']:
            frame_bgr = np.ascontiguousarray(frame_bgr)
        self.proc.stdin.write(frame_bgr.tobytes())
```

`write_pred_video` 對應改寫：

```python
def write_pred_video(video_file, pred_dict, save_file, traj_len=8,
                     label_df=None, codec='h264_nvenc'):
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)        # ← 不再 int()，保留 float
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ...
    # 用 FFmpegWriter；失敗則 fallback 到 libx264
    try:
        out = FFmpegWriter(save_file, w, h, fps, codec=codec)
        used_codec = codec
    except Exception as e:
        print(f'[write_pred_video] {codec} init failed ({e}), fallback to libx264')
        out = FFmpegWriter(save_file, w, h, fps, codec='libx264')
        used_codec = 'libx264'

    n_pred = len(x_pred)                   # ← 新增邊界檢查
    i = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        if i >= n_pred:                    # ← 預測用完就停
            break
        ...
```

順便修掉的兩個小 bug：

- `fps` 保留 float
- 用 `n_pred` 限制迴圈，影片比預測長時不再 IndexError

支援的 codec：

| codec        | 用途              | 速度  | 相容性     |
|--------------|-------------------|-------|------------|
| `h264_nvenc` | NVIDIA GPU 硬編   | 最快  | 需 NVENC   |
| `libx264`    | CPU 軟編          | 慢    | 最好       |

---

## Phase 2：Median 影像產生加速

對應 `Video_IterableDataset.__gen_median__`。

**Before（原版）**

```python
def __gen_median__(self, max_sample_num, video_range):
    print('Generate median image...')
    if video_range is None:
        start_frame, end_frame = 0, self.video_len
    else:
        start_frame = max(0, video_range[0] * self.fps)
        end_frame = min(video_range[1] * self.fps, self.video_len)
    video_seg_len = end_frame - start_frame

    if video_seg_len > max_sample_num:
        sample_step = video_seg_len // max_sample_num
    else:
        sample_step = 1

    frame_list = []
    for i in range(start_frame, end_frame, sample_step):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)   # ← 每次都 seek
        success, frame = self.cap.read()
        if not success:
            break
        frame_list.append(frame)                   # ← 原始解析度
    median = np.median(frame_list, 0)[..., ::-1]
    if self.bg_mode == 'concat':
        median = Image.fromarray(median.astype('uint8'))
        median = np.array(median.resize(size=(self.WIDTH, self.HEIGHT)))
        median = np.moveaxis(median, -1, 0)
    print('Median image generated.')
    return median
```

問題：

1. **每次取樣都 `cap.set(CAP_PROP_POS_FRAMES, i)`**：在大部分 codec 上 seek 操作都需要 demux 到最近的 keyframe，`O(N/sample_step)` 次 seek 反而比順序讀整段慢。
2. **frame 以原始解析度（例如 4K）存進 list**：1800 個 4K 幀約 25 GB RAM。
3. **`np.median` 在原始解析度上算**：4K × 1800 → bottleneck。
4. **沒有計時觀察**。

**After（新版）**

```python
def __gen_median__(self, max_sample_num, video_range):
    print('Generate median image...')
    t_total = time.perf_counter()

    if video_range is None:
        start_frame, end_frame = 0, self.video_len
    else:
        start_frame = max(0, video_range[0] * self.fps)
        end_frame = min(video_range[1] * self.fps, self.video_len)
    video_seg_len = end_frame - start_frame
    sample_step = max(1, video_seg_len // max_sample_num)

    cap = cv2.VideoCapture(self.video_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # ← 只 seek 一次

    # 順序讀：grab() 跳過不要的幀，只 decode 需要的
    n_sample = (end_frame - start_frame + sample_step - 1) // sample_step
    small_frames = np.empty((n_sample, self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)

    t_read = time.perf_counter()
    idx = 0
    cur = start_frame
    while cur < end_frame and idx < n_sample:
        # 要 decode 的幀
        success, frame = cap.read()
        if not success:
            break
        small_frames[idx] = cv2.resize(frame, (self.WIDTH, self.HEIGHT),
                                       interpolation=cv2.INTER_LINEAR)
        idx += 1
        cur += 1

        # 跳過 sample_step - 1 幀（只 demux，不 decode）
        for _ in range(sample_step - 1):
            if cur >= end_frame:
                break
            if not cap.grab():
                break
            cur += 1

    cap.release()
    small_frames = small_frames[:idx]
    print(f'  [median] read+resize {idx} frames: {time.perf_counter()-t_read:.2f}s')

    t_med = time.perf_counter()
    median = np.median(small_frames, axis=0).astype(np.uint8)[..., ::-1]
    print(f'  [median] np.median: {time.perf_counter()-t_med:.2f}s')

    if self.bg_mode == 'concat':
        median = np.moveaxis(median, -1, 0)

    print(f'Median image generated. (total {time.perf_counter()-t_total:.2f}s)')
    return median
```

四個關鍵改動：

| 改動 | Before | After |
|------|--------|-------|
| Seek 策略 | 每幀 `cap.set(POS_FRAMES, i)` | 起點 seek 一次後順序 `grab()` |
| 不需要的幀 | `cap.read()` 全部 decode 後丟掉 | `cap.grab()` 只 demux 不 decode |
| 取樣解析度 | 原始解析度（如 4K） | 預先 resize 到 model input（512×288）再存 |
| Buffer | `list` + `np.median(frame_list)` | `np.empty(n_sample, H, W, 3)` 預先 alloc |
| 觀察 | 無 | 分階段印 read+resize / np.median 各自耗時 |

`concat` 模式下 median 已經以小尺寸存好，外面 `_get_median_chw()` 只要 transpose 不需 resize。

`_preprocess_one_frame` 用的 `_median_hwc_i16` / `_median_chw_f32` 在 `__init__` 一次計算後快取，`__iter__` 主迴圈每幀都不再重複轉換。

---

## Phase 3：`draw_traj` 改用 OpenCV in-place 繪製

**Before（原版）**

```python
def draw_traj(img, traj, radius=3, color='red'):
    """ Draw trajectory on the image. """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)                              # numpy → PIL，一次 copy

    for i in range(len(traj)):
        if traj[i] is not None:
            draw_x = traj[i][0]
            draw_y = traj[i][1]
            bbox = (draw_x - radius, draw_y - radius,
                    draw_x + radius, draw_y + radius)
            draw = ImageDraw.Draw(img)                      # 每點都重建 Draw 物件
            draw.ellipse(bbox, fill='rgb(255,255,255)', outline=color)
            del draw
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)    # PIL → numpy，再一次 copy
    return img
```

問題：

1. 每幀兩次完整 frame 的 memcpy（`numpy → PIL`、`PIL → numpy`）。
2. 額外兩次 `cv2.cvtColor`（BGR↔RGB）也都會 copy。
3. `ImageDraw.Draw(img)` 在迴圈中為每個點重建 Draw 物件。
4. PIL 繪圖 backend 比 OpenCV 慢一個量級。

**After（新版）**

加入顏色查表 + 純 OpenCV、in-place 繪製：

```python
_COLOR_MAP = {
    'red':    (0, 0, 255),
    'yellow': (0, 255, 255),
    'green':  (0, 255, 0),
    'blue':   (255, 0, 0),
    'white':  (255, 255, 255),
    'black':  (0, 0, 0),
}

def draw_traj(img, traj, radius=3, color='red'):
    """Draw trajectory on the image (in-place, OpenCV only)."""
    outline_bgr = _COLOR_MAP.get(color, (0, 0, 255))
    fill_bgr = (255, 255, 255)
    for p in traj:
        if p is None:
            continue
        x, y = int(p[0]), int(p[1])
        # 先畫白色實心填充
        cv2.circle(img, (x, y), radius, fill_bgr, thickness=-1, lineType=cv2.LINE_AA)
        # 再畫彩色外框
        cv2.circle(img, (x, y), radius, outline_bgr, thickness=1, lineType=cv2.LINE_AA)
    return img
```

關鍵改變：

- **零 numpy ↔ PIL 轉換**：`cv2.circle` 直接修改原 frame buffer。
- **零 BGR ↔ RGB 轉換**：直接用 BGR 值。
- **抗鋸齒一次到位**：`cv2.LINE_AA`，無需後處理。
- **白底 + 彩色外框**：在球場深淺背景下都清楚可辨。

對 `write_pred_video` 而言，每幀的繪圖開銷從原本的「2 次 frame-sized memcpy + 2 次 cvtColor + N 次 PIL ellipse」減為「2N 次 cv2.circle」。

---

## 改動總表

| Phase | 檔案 | 修改項目 | 改動類型 |
|-------|------|----------|----------|
| 1 | `dataset.py` | `Video_IterableDataset.__iter__` | 重寫為 deque 串流 |
| 1 | `dataset.py` | `Video_IterableDataset._preprocess_one_frame` | 新增 |
| 1 | `dataset.py` | `Video_IterableDataset._assemble` | 新增 |
| 1 | `dataset.py` | `Video_IterableDataset.__process__` | PIL 迴圈 → numpy 向量化 |
| 1 | `dataset.py` | `_get_median_hwc` / `_get_median_chw` | 新增 |
| 1 | `dataset.py` | `Video_IterableDataset.__init__` 預先快取 median | 新增 |
| 1 | `utils/general.py` | `FFmpegWriter` | 新增 |
| 1 | `utils/general.py` | `write_pred_video` | 改用 FFmpegWriter，加 codec 參數與 fallback，修 fps/邊界 bug |
| 2 | `dataset.py` | `Video_IterableDataset.__gen_median__` | 改 `grab()` 跳幀 + 預先 alloc + 預先 resize + 計時 |
| 3 | `utils/general.py` | `_COLOR_MAP` | 新增 |
| 3 | `utils/general.py` | `draw_traj` | PIL 雙向 memcpy → cv2 in-place |

---

## 預期效益

三個 phase 各自鎖定獨立瓶頸：

- **Phase 1** — 消除 sliding window 的冗餘前處理（`O(N×L) → O(N)`）；影片寫出從 CPU mp4v 改為 NVENC 硬編。
- **Phase 2** — 長影片 median 初始化從「每幀 seek + 全解析度 decode」改為「順序 grab + 取樣後立即 resize」，記憶體與時間皆顯著下降。
- **Phase 3** — 繪圖端從 PIL 雙向 memcpy 改為 cv2 in-place，輸出視覺化幾乎不再是 wall time 占比要角。

完成後，CPU 端的前處理、median init、影片繪製/寫出都已最小化。下一階段（Phase 4）才有條件透過 `non_blocking` device transfer + pinned memory 把 CPU↔GPU 的同步點消掉，那部分不在本文件範圍。
