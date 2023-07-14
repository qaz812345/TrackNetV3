# TrackNetV3

## Installation
* Develop Environment
    ```
    Ubuntu 16.04.7 LTS
    Python 3.8.7
    torch 1.10.0
    ```
* Clone this reposity.

    `git clone https://github.com/qaz812345/TrackNetV3.git`

* Install the requirements.

    `pip install -r requirements.txt`

## Inference
* Prediction

    `python3 predict.py --video_file test.mp4 --model_file model_best.pt --save_dir prediction`

* Show trajectory

    `python3 show_trajectory.py --video_file test.mp4 --csv_file prediction/test_ball.csv --save_dir prediction`

## Training
### Prepare Dataset
* Shuttlecock Trajectory Dataset description: https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw 
* Set the data root directory in ```dataset.py```.
* Exclude `train/match16` due to inconsistent camera viewpoints.
* Data Preprocessing

    `python preprocess.py`
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

    `python train.py --model_name TrackNet --seq_len 8 --epochs 30 --batch_size 10 --bg_mode concat --alpha 0.5 --save_dir exp`

* Resume training (start from last epoch)

    `python train.py --epochs 30 --save_dir exp --resume_training`

### Inpainting Mask Generation
`python predict_dataset.py --batch_size 16 --model_file TrackNet_best.pt`

### Train InpaintNet
`python train.py --model_name InpaintNet --seq_len 16 --epoch 300 --batch_size 32 --mask_ratio 0.3 --save_dir exp`

## Evaluation
* Evaluation on test set

    `python evaluation.py --batch_size 16 --model_file TrackNet_best.pt --save_dir eval`

* Show predicted video with label

    `python show_rally.py --frame_dir <frame_dir>`

## Error Analysis Interface
`python error_analysis.py --split test`