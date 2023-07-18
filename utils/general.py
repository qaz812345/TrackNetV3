import os
import cv2
import math
import json
import parse
import shutil
import numpy as np
import pandas as pd

from collections import deque
from PIL import Image, ImageDraw
from model import TrackNet, InpaintNet

HEIGHT = 288
WIDTH = 512
DELTA_T = 1/math.sqrt(HEIGHT**2 + WIDTH**2)
COOR_TH = DELTA_T * 50

class ResumeArgumentParser():
    """ A simple argument parser for parsing the configuration file."""
    def __init__(self, param_dict):
        self.model_name = param_dict['model_name']
        self.seq_len = param_dict['seq_len']
        self.epochs = param_dict['epochs']
        self.batch_size = param_dict['batch_size']
        self.optim = param_dict['optim']
        self.learning_rate = param_dict['learning_rate']
        self.lr_scheduler = param_dict['lr_scheduler']
        self.bg_mode = param_dict['bg_mode']
        self.alpha = param_dict['alpha']
        self.frame_alpha = param_dict['frame_alpha']
        self.mask_ratio = param_dict['mask_ratio']
        self.tolerance = param_dict['tolerance']
        self.resume_training = param_dict['resume_training']
        self.seed = param_dict['seed']
        self.save_dir = param_dict['save_dir']
        self.debug = param_dict['debug']
        self.verbose = param_dict['verbose']


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

def generate_frames(video_file):
    assert video_file[-4:] == '.mp4'
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_list = []
    success = True

    # Sample frames until video end or exceed the number of labels
    while success:
        success, frame = cap.read()
        if success:
            frame_list.append(frame)
            
    return frame_list, fps, (w, h)

def write_pred_video(frame_list, video_cofig, pred_dict, save_file, traj_len=8, label_df=None):
    if label_df is not None:
        f_i, x, y, vis = label_df['Frame'], label_df['X'], label_df['Y'], label_df['Visibility']
    
    x_pred, y_pred, vis_pred = pred_dict['X'], pred_dict['Y'], pred_dict['Visibility']

    # Video config
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_file, fourcc, video_cofig['fps'], video_cofig['shape'])
    
    # For storing trajectory
    pred_queue = deque()
    if label_df is not None:
        gt_queue = deque()
    
    # Draw prediction
    for i, frame in enumerate(frame_list):
        # Check capacity of queue
        if len(pred_queue) >= traj_len:
            pred_queue.pop()
        if label_df is not None and len(gt_queue) >= traj_len:
            gt_queue.pop()
        
        # Push ball coordinates for each frame
        if label_df is not None:
            gt_queue.appendleft([x[i], y[i]]) if vis[i] and i < len(label_df) else gt_queue.appendleft(None)
        pred_queue.appendleft([x_pred[i], y_pred[i]]) if vis_pred[i] else pred_queue.appendleft(None)

        # Convert to PIL image for drawing
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   
        img = Image.fromarray(img)

        # Draw ball trajectory
        if label_df is not None:
            for i in range(len(gt_queue)):
                if gt_queue[i] is not None:
                    draw_x = gt_queue[i][0]
                    draw_y = gt_queue[i][1]
                    bbox =  (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
                    draw = ImageDraw.Draw(img)
                    draw.ellipse(bbox, outline ='red')
        
        for i in range(len(pred_queue)):
            if pred_queue[i] is not None:
                draw_x = pred_queue[i][0]
                draw_y = pred_queue[i][1]
                bbox =  (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
                draw = ImageDraw.Draw(img)
                draw.ellipse(bbox, outline ='yellow')
                del draw

        # Convert back to cv2 image and write to output video
        frame =  cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()

def write_pred_csv(pred_dict, save_file=None, save_inpaint_mask=False):
    if save_inpaint_mask:
        pred_df = pd.DataFrame({'Frame': pred_dict['Frame'], 'Visibility': pred_dict['Visibility'], 'X': pred_dict['X'], 'Y': pred_dict['Y'], 'Inpainting': pred_dict['Inpainting']})
    else:
        pred_df = pd.DataFrame({'Frame': pred_dict['Frame'], 'Visibility': pred_dict['Visibility'], 'X': pred_dict['X'], 'Y': pred_dict['Y']})
    pred_df.to_csv(save_file, index=False)
    
############################### Preprocessing Functions ###############################
def generate_data_frames(video_file):
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
    median = median[..., ::-1] # BGR to RGB
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
    save_dir = os.path.join(match_dir, 'frame', rally_id)
    frames = []
    cap = cv2.VideoCapture(video_file)
    success = True
    while success:
        success, frame = cap.read()
        if success:
            frames.append(frame)
    
    median = np.median(np.array(frames), 0)[..., ::-1]
    np.savez(os.path.join(save_dir, 'median.npz'), median=median) # must be lossless    