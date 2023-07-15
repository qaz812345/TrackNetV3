# TrackNetV3

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
* Predict the label csv from the video
    ```
    python3 predict.py --video_file test.mp4 --tracknet_file TrackNet_best.pt --inpaintnet_file InpaintNet_best.pt --save_dir prediction
    ```

* Predict the label csv from the video, and output a video with predicted trajectory
    ```
    python3 predict.py --video_file test.mp4 --tracknet_file TrackNet_best.pt --inpaintnet_file InpaintNet_best.pt --save_dir prediction --output_video
    ```

## Training
### Prepare Dataset
* Shuttlecock Trajectory Dataset description: https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw 
* Set the data root directory in ```dataset.py```.
* Exclude `train/match16` due to inconsistent camera viewpoints.
* Data Preprocessing
    ```
    python preprocess.py
    ```
* The `frame` directories and the `val` directory will be generated after preprocessing.
* Dataset File Structure:
```
Shuttlecock_Trajectory_Dataset
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
    |   ├── match15/
    |   ├── match24/
    |   ├── match25/
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

### Train TrackNet
* Train from scratch 
    ```
    python train.py --model_name TrackNet --seq_len 8 --epochs 30 --batch_size 10 --bg_mode concat --alpha 0.5 --save_dir exp  --verbose
    ```

* Resume training (start from the last epoch to the specified epoch)
    ```
    python train.py --model_name TrackNet --epochs 30 --save_dir exp --resume_training
    ```

### Inpainting Mask Generation
* Generate predicted trajectories and inpainting mask
    ```
    python generate_mask_data.py --model_file TrackNet_best.pt --batch_size 16
    ```

### Train InpaintNet
* Train from scratch 
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
    python test.py --tracknet_file TrackNet_best.pt --inpaintnet_file InpaintNet_best.pt --split test --batch_size 16 --eval_mode weight --save_dir eval
    ```

* Evaluate TrackNetV3 on test set and save the detail results for error analysis
    ```
    python test.py --tracknet_file TrackNet_best.pt --inpaintnet_file InpaintNet_best.pt --split test --batch_size 16 --eval_mode weight --save_dir eval --output_pred
    ```

* Show predicted video with ground truth label
    ```
    python test.py --video_file <video_file>
    ```

## Error Analysis Interface
* A simple Dash application
    ```
    python error_analysis.py --split test --host 127.0.0.1
    ```