import os
import cv2
import math
import json
import parse
import shutil
import numpy as np
import pandas as pd

from model import TrackNet, InpaintNet

HEIGHT = 288
WIDTH = 512

###################################  Helper Functions ###################################
def get_model(model_name, seq_len=None, bg_mode=None):
    """ Create model by name and the configuration parameter.

        args:
            model_name - A str of model name
            seq_len - An int specifying the length of a single input sequence
            bg_mode - A str specifying the background mode, '', 'subtract', 'subtract_concat', 'concat'

        returns:
            model - A torch.nn.Module
    """
    if model_name == 'TrackNet':
        if bg_mode == 'subtract':
            model = TrackNet(in_dim=seq_len, out_dim=seq_len)
        elif bg_mode == 'subtract_concat':
            model = TrackNet(in_dim=seq_len*4, out_dim=seq_len)
        elif bg_mode == 'concat':
            model = TrackNet(in_dim=(seq_len+1)*3, out_dim=seq_len)
        else:
            model = TrackNet(in_dim=seq_len*3, out_dim=seq_len)
    elif model_name == 'InpaintNet':
        model = InpaintNet()
    else:
        raise ValueError('Invalid model name.')
    
    return model

def list_dirs(directory):
    """ Extension of os.listdir which return the directory pathes including input directory.

        args:
            directory - A str of directory path

        returns:
            A list of directory pathes
    """
    return sorted([os.path.join(directory, path) for path in os.listdir(directory)])

def to_img(image):
    """ Convert the normalized image back to RGB image.

        args:
            image - A numpy.ndarray of images with range [0, 1].

        returns:
            image - A numpy.ndarray of images with range [0, 255].
    """
    image = image * 255
    image = image.astype('uint8')
    return image

def to_img_format(input, num_ch=1):
    """ Helper function for transforming torch input format to image sequence format.

        args:
            input - A numpy.ndarray with shape (N, F*C, H, W)
            num_ch - A int specifying the number of channels of each frame

        returns:
            A numpy.ndarray of image sequences with shape (N, F, H, W) or (N, F, H, W, 3)
    """
    assert len(input.shape) > 3
    
    if num_ch == 1:
        # (N, F, H ,W)
        return input
    else:
        # (N, F*C, H ,W)
        input = np.transpose(input, (0, 2, 3, 1)) # (N, H ,W, F*C)
        seq_len = int(input.shape[-1]/num_ch)
        img_seq = np.array([]).reshape(0, seq_len, HEIGHT, WIDTH, 3)
        for n in range(input.shape[0]):
            frame = np.array([]).reshape(0, HEIGHT, WIDTH, 3)
            for f in range(0, input.shape[-1], num_ch):
                img = input[n, :, :, f:f+3]
                frame = np.concatenate((frame, img.reshape(1, HEIGHT, WIDTH, 3)), axis=0)
            img_seq = np.concatenate((img_seq, frame.reshape(1, seq_len, HEIGHT, WIDTH, 3)), axis=0)
        
        return img_seq

def get_num_frames(video_file):
    """ Return the number of frames in the video.

        args:
            video_file - A str of video file path with format <data_dir>/<split>/match<match_id>/video/<rally_id>.mp4

        returns:
            A int specifying the number of frames in the video
    """
    assert video_file[-4:] == '.mp4'
    match_dir, rally_id = parse.parse('{}/video/{}.mp4', video_file)
    rally_dir = os.path.join(match_dir, 'frame', rally_id)
    assert os.path.exists(rally_dir)

    return len(os.listdir(rally_dir)) - 1

def get_rally_dirs(data_dir, split):
    match_dirs = os.listdir(os.path.join(data_dir, split))
    match_dirs = [os.path.join(split, d) for d in match_dirs]
    match_dirs = sorted(match_dirs, key=lambda s: int(s.split('match')[-1]))
    rally_dirs = []
    for match_dir in match_dirs:
        rally_dir = os.listdir(os.path.join(data_dir, match_dir, 'frame'))
        rally_dir = sorted(rally_dir)
        rally_dir = [os.path.join(match_dir, 'frame', d) for d in rally_dir]
        rally_dirs.extend(rally_dir)
    
    return rally_dirs # split/match#/frame/rally_id

############################### Preprocessing Functions ###############################
def generate_frames(video_file):
    """ Sample frames from the video.

        args:
            video_file - A str of video file path with format <data_dir>/<split>/match<match_id>/video/<rally_id>.mp4
    """
    assert video_file[-4:] == '.mp4'
    try:
        match_dir, rally_id = parse.parse('{}/video/{}.mp4', video_file)
        csv_file = os.path.join(match_dir, 'csv', f'{rally_id}_ball.csv')
        assert os.path.exists(video_file) and os.path.exists(csv_file)
    except:
        print(f'{video_file} has no matched csv file.')
        return

    rally_dir = os.path.join(match_dir, 'frame', rally_id)
    if not os.path.exists(rally_dir):
        # Haven't process
        os.makedirs(rally_dir)
    else:
        label_df = pd.read_csv(csv_file, encoding='utf8')
        if len(list_dirs(rally_dir)) < len(label_df):
            # Some error has occured, process again
            shutil.rmtree(rally_dir)
            os.makedirs(rally_dir)
        else:
            # Already processed.
            return

    label_df = pd.read_csv(csv_file, encoding='utf8')
    cap = cv2.VideoCapture(video_file)
    frames = []
    success = True

    # Sample frames until video end or exceed the number of labels
    while success and len(frames) != len(label_df):
        success, frame = cap.read()
        if success:
            frames.append(frame)
            cv2.imwrite(os.path.join(rally_dir, f'{len(frames)-1}.png'), frame)#, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    median = np.median(np.array(frames), 0)
    np.savez(os.path.join(rally_dir, 'median.npz'), median=median) # must be lossless

def get_match_median(match_dir):
    """ Calculate the match median frame.

        args:
            match_dir - A str of match directory path with format <data_dir>/<split>/match<match_id>
    """
    medians = []
    rally_dirs = list_dirs(os.path.join(match_dir, 'frame'))
    for rally_dir in rally_dirs:
        _, rally_id = parse.parse('{}/frame/{}', rally_dir)
        if not os.path.exists(os.path.join(rally_dir, 'median.npz')):
            get_rally_median(os.path.join(match_dir, 'video', f'{rally_id}.mp4'))
        frame = np.load(os.path.join(rally_dir, 'median.npz'))['median']
        medians.append(frame)
    
    median = np.median(np.array(medians), 0)
    np.savez(os.path.join(match_dir, 'median.npz'), median=median) # must be lossless

def get_rally_median(video_file):
    """ Calculate the rally median frame.

        args:
            video_file - A str of video file path with format <data_dir>/<split>/match<match_id>/video/<rally_id>.mp4
    """
    match_dir, rally_id = parse.parse('{}/video/{}.mp4', video_file)
    
    frames = []
    cap = cv2.VideoCapture(video_file)
    success = True
    while success:
        success, frame = cap.read()
        if success:
            frames.append(frame)
    
    median = np.median(np.array(frames), 0)
    np.savez(f'{save_dir}/median.npz', median=median) # must be lossless    
