# TrackNetV3
## Inference

* Prediction

    `python3 predict.py --video_file test.mp4 --model_file model_best.pt --save_dir prediction`

* Show trajectory

    `python3 show_trajectory.py --video_file test.mp4 --csv_file prediction/test_ball.csv --save_dir prediction`

## Training
### Prepare Dataset
* Shuttlecock Trajectory Dataset description: https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw 
* Set the data root directory in ```dataset.py```.
* Dataset File Structure:
```
Shuttlecock_Trajectory_Dataset
    ├─ train
    |    ├── match1/
    |    │     ├── csv/
    |    │     │     ├── 1_01_00_ball.csv
    |    │     │     ├── 1_02_00_ball.csv
    |    │     │     ├── …
    |    │     │     └── *_**_**_ball.csv
    |    │     ├── frame/
    |    │     │     ├── 1_01_00/
    |    │     │     │     ├── 0.png
    |    │     │     │     ├── 1.png
    |    │     │     │     ├── …
    |    │     │     │     └── *.png
    |    │     │     ├── 1_02_00/
    |    │     │     │     ├── 0.png
    |    │     │     │     ├── 1.png
    |    │     │     │     ├── …
    |    │     │     │     └── *.png
    |    │     │     ├── …
    |    │     │     └── *_**_**/
    |    │     │
    |    │     └── video/
    |    │           ├── 1_01_00.mp4
    |    │           ├── 1_02_00.mp4
    |    │           ├── …
    |    │           └── *_**_**.mp4
    |    ├── match2/
    |    │ ⋮
    |    ├── match15/
    |    ├── match24/
    |    ├── match25/
    |    └── match26/
    |
    └─ test
            ├── match1/
            ├── match2/
            └── match3/
```

* Sample frames from videos.

    `python preprocess.py`

* Replacing the files in the `csv` directory with the corresponding files from the `correct_csv` directory.


### Train TrackNet
* Train from scratch 

    `python train.py --num_frame 8 --epochs 30 --batch_size 10 --save_dir exp`

* Resume training (start from last epoch)

    `python3 train.py --epochs 30 --save_dir exp --resume_training`

### Inpainting Mask Generation
`python predict_dataset.py --batch_size 16 --model_file model_best.pt`

### Train InpaintNet
`python train.py --num_frame 16 --epoch 300 --batch_size 32 --save_dir exp`

## Evaluation
* Evaluation on test set

    `python3 evaluation.py --batch_size 16 --model_file model_best.pt --save_dir eval`

* Show predicted video with label

    `python3 show_rally.py --frame_dir <frame_dir> --model_file model_best.pt --batch_size 16 --output_mode both --save_dir eval`
