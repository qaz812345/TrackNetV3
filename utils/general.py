import os
import cv2
import json
import math
import parse
import shutil
import numpy as np
import pandas as pd

from collections import deque
from PIL import Image, ImageDraw
from model import TrackNet, InpaintNet

# Global variables
HEIGHT = 288
WIDTH = 512
SIGMA = 2.5
DELTA_T = 1/math.sqrt(HEIGHT**2 + WIDTH**2)
COOR_TH = DELTA_T * 50
IMG_FORMAT = 'png'


class ResumeArgumentParser():
    """ A argument parser for parsing the parameter dictionary from checkpoint file."""
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

        Args:
            model_name (str): type of model to create
                Choices:
                    - 'TrackNet': Return TrackNet model
                    - 'InpaintNet': Return InpaintNet model
            seq_len (int, optional): Length of input sequence of TrackNet
            bg_mode (str, optional): Background mode of TrackNet
                Choices:
                    - '': Return TrackNet with L x 3 input channels (RGB)
                    - 'subtract': Return TrackNet with L x 1 input channel (Difference frame)
                    - 'subtract_concat': Return TrackNet with L x 4 input channels (RGB + Difference frame)
                    - 'concat': Return TrackNet with (L+1) x 3 input channels (RGB)

        Returns:
            model (torch.nn.Module): Model with specified configuration
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

def show_model_size(model):
    """ Estimate the size of the model.
        reference: https://discuss.pytorch.org/t/finding-model-size/130275/2

        Args:
            model (torch.nn.Module): target model
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f'Model size: {size_all_mb:.3f}MB')

def list_dirs(directory):
    """ Extension of os.listdir which return the directory pathes including input directory.

        Args:
            directory (str): Directory path

        Returns:
            (List[str]): Directory pathes with pathes including input directory
    """

    return sorted([os.path.join(directory, path) for path in os.listdir(directory)])

def to_img(image):
    """ Convert the normalized image back to image format.

        Args:
            image (numpy.ndarray): Images with range in [0, 1]

        Returns:
            image (numpy.ndarray): Images with range in [0, 255]
    """

    image = image * 255
    image = image.astype('uint8')
    return image

def to_img_format(input, num_ch=1):
    """ Helper function for transforming model input sequence format to image sequence format.

        Args:
            input (numpy.ndarray): model input with shape (N, L*C, H, W)
            num_ch (int): Number of channels of each frame.

        Returns:
            (numpy.ndarray): Image sequences with shape (N, L, H, W) or (N, L, H, W, 3)
    """

    assert len(input.shape) == 4, 'Input must be 4D tensor.'
    
    if num_ch == 1:
        # (N, L, H ,W)
        return input
    else:
        # (N, L*C, H ,W)
        input = np.transpose(input, (0, 2, 3, 1)) # (N, H ,W, L*C)
        seq_len = int(input.shape[-1]/num_ch)
        img_seq = np.array([]).reshape(0, seq_len, HEIGHT, WIDTH, 3) # (N, L, H, W, 3)
        # For each sample in the batch
        for n in range(input.shape[0]):
            frame = np.array([]).reshape(0, HEIGHT, WIDTH, 3)
            # Get each frame in the sequence
            for f in range(0, input.shape[-1], num_ch):
                img = input[n, :, :, f:f+3]
                frame = np.concatenate((frame, img.reshape(1, HEIGHT, WIDTH, 3)), axis=0)
            img_seq = np.concatenate((img_seq, frame.reshape(1, seq_len, HEIGHT, WIDTH, 3)), axis=0)
        
        return img_seq

def get_num_frames(rally_dir):
    """ Return the number of frames in the video.

        Args:
            rally_dir (str): File path of the rally frame directory 
                Format: '{data_dir}/{split}/match{match_id}/frame/{rally_id}'

        Returns:
            (int): Number of frames in the rally frame directory
    """

    try:
        frame_files = list_dirs(rally_dir)
    except:
        raise ValueError(f'{rally_dir} does not exist.')
    frame_files = [f for f in frame_files if f.split('.')[-1] == IMG_FORMAT]
    return len(frame_files)

def get_rally_dirs(data_dir, split):
    """ Return all rally directories in the split.

        Args:
            data_dir (str): File path of the data root directory
            split (str): Split name

        Returns:
            rally_dirs: (List[str]): Rally directories in the split
                Format: ['{split}/match{match_id}/frame/{rally_id}', ...]
    """

    rally_dirs = []

    # Get all match directories in the split
    match_dirs = os.listdir(os.path.join(data_dir, split))
    match_dirs = [os.path.join(split, d) for d in match_dirs]
    match_dirs = sorted(match_dirs, key=lambda s: int(s.split('match')[-1]))
    
    # Get all rally directories in the match directory
    for match_dir in match_dirs:
        rally_dir = os.listdir(os.path.join(data_dir, match_dir, 'frame'))
        rally_dir = sorted(rally_dir)
        rally_dir = [os.path.join(match_dir, 'frame', d) for d in rally_dir]
        rally_dirs.extend(rally_dir)
    
    return rally_dirs

def generate_frames(video_file):
    """ Sample frames from the video.

        Args:
            video_file (str): File path of the video file

        Returns:
            frame_list (List[numpy.ndarray]): List of sampled frames
    """

    assert video_file[-4:] == '.mp4', 'Invalid video file format.'

    # Get camera parameters
    cap = cv2.VideoCapture(video_file)
    frame_list = []
    success = True

    # Sample frames until video end
    while success:
        success, frame = cap.read()
        if success:
            frame_list.append(frame)
            
    return frame_list

def draw_traj(img, traj, radius=3, color='red'):
    """ Draw trajectory on the image.

        Args:
            img (numpy.ndarray): Image with shape (H, W, C)
            traj (deque): Trajectory to draw

        Returns:
            img (numpy.ndarray): Image with trajectory drawn
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
    img = Image.fromarray(img)
    
    for i in range(len(traj)):
        if traj[i] is not None:
            draw_x = traj[i][0]
            draw_y = traj[i][1]
            bbox =  (draw_x - radius, draw_y - radius, draw_x + radius, draw_y + radius)
            draw = ImageDraw.Draw(img)
            draw.ellipse(bbox, fill='rgb(255,255,255)', outline=color)
            del draw
    img =  cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    return img

def write_pred_video(video_file, pred_dict, save_file, traj_len=8, label_df=None):
    """ Write a video with prediction result.

        Args:
            video_file (str): File path of the input video file
            pred_dict (Dict): Prediction result
                Format: {'Frame': frame_id (List[int]),
                         'X': x_pred (List[int]),
                         'Y': y_pred (List[int]),
                         'Visibility': vis_pred (List[int])}
            save_file (str): File path of the output video file
            traj_len (int, optional): Length of trajectory to draw
            label_df (pandas.DataFrame, optional): Ground truth label dataframe
        
        Returns:
            None
    """
    # Read video
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    # Read ground truth label if exists
    if label_df is not None:
        f_i, x, y, vis = label_df['Frame'], label_df['X'], label_df['Y'], label_df['Visibility']
    
    # Read prediction result
    x_pred, y_pred, vis_pred = pred_dict['X'], pred_dict['Y'], pred_dict['Visibility']

    # Video config
    out = cv2.VideoWriter(save_file, fourcc, fps, (w, h))
    
    # Create a queue for storing trajectory
    pred_queue = deque()
    if label_df is not None:
        gt_queue = deque()
    
    # Draw label and prediction trajectory
    #for i, frame in enumerate(frame_list):
    i = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Check capacity of queue
        if len(pred_queue) >= traj_len:
            pred_queue.pop()
        if label_df is not None and len(gt_queue) >= traj_len:
            gt_queue.pop()
        
        # Push ball coordinates for each frame
        if label_df is not None:
            gt_queue.appendleft([x[i], y[i]]) if vis[i] and i < len(label_df) else gt_queue.appendleft(None)
        pred_queue.appendleft([x_pred[i], y_pred[i]]) if vis_pred[i] else pred_queue.appendleft(None)

        # Draw ground truth trajectory if exists
        if label_df is not None:
            frame = draw_traj(frame, gt_queue, color='red')
        
        # Draw prediction trajectory
        frame = draw_traj(frame, pred_queue, color='yellow')

        out.write(frame)
        i+=1

    out.release()
    cap.release()

def write_pred_csv(pred_dict, save_file, save_inpaint_mask=False):
    """ Write prediction result to csv file.

        Args:
            pred_dict (Dict): Prediction result
                Format: {'Frame': frame_id (List[int]),
                         'X': x_pred (List[int]),
                         'Y': y_pred (List[int]),
                         'Visibility': vis_pred (List[int]),
                         'Inpaint_Mask': inpaint_mask (List[int])}
            save_file (str): File path of the output csv file
            save_inpaint_mask (bool, optional): Whether to save inpaint mask

        Returns:
            None
    """

    if save_inpaint_mask:
        # Save temporary data for InpaintNet training
        pred_df = pd.DataFrame({'Frame': pred_dict['Frame'],
                                'Visibility_GT': pred_dict['Visibility_GT'],
                                'X_GT': pred_dict['X_GT'],
                                'Y_GT': pred_dict['Y_GT'],
                                'Visibility': pred_dict['Visibility'],
                                'X': pred_dict['X'], 
                                'Y': pred_dict['Y'],
                                'Inpaint_Mask': pred_dict['Inpaint_Mask']})
    else:
        pred_df = pd.DataFrame({'Frame': pred_dict['Frame'],
                                'Visibility': pred_dict['Visibility'],
                                'X': pred_dict['X'],
                                'Y': pred_dict['Y']})
    pred_df.to_csv(save_file, index=False)
    
def convert_gt_to_coco_json(data_dir, split, drop=False):
    """ Convert ground truth csv file to coco format json file.

        Args:
            split (str): Split name
        
        Returns:
            None
    """
    if split == 'test' and drop:
        drop_frame_dict = json.load(open(os.path.join(data_dir, 'drop_frame.json')))
        start_frame, end_frame = drop_frame_dict['start'], drop_frame_dict['end']
    bbox_size = 10
    rally_dirs = get_rally_dirs(data_dir, split)
    rally_dirs = [os.path.join(data_dir, rally_dir) for rally_dir in rally_dirs]
    image_info = []
    annotations = []
    sample_count = 0
    for rally_dir in rally_dirs:
        file_format_str = os.path.join('{}', 'frame', '{}')
        match_dir, rally_id = parse.parse(file_format_str, rally_dir)
        match_id = match_dir.split('match')[-1]
        csv_file = os.path.join(match_dir, 'corrected_csv', f'{rally_id}_ball.csv') if split == 'test' else os.path.join(match_dir, 'csv', f'{rally_id}_ball.csv')
        label_df = pd.read_csv(csv_file, encoding='utf8')
        f, x, y, v = label_df['Frame'].values, label_df['X'].values, label_df['Y'].values, label_df['Visibility'].values
        if split == 'test' and drop:
            rally_key = f'{match_id}_{rally_id}'
            start_f, end_f = start_frame[rally_key], end_frame[rally_key]
            f, x, y, v = f[start_f:end_f] ,x[start_f:end_f], y[start_f:end_f], v[start_f:end_f]
        w, h = Image.open(f'{match_dir}/frame/{rally_id}/0.{IMG_FORMAT}').size
        for i, cx, cy, vis in zip(f, x, y, v):
            image_info.append({'id': sample_count, 'width': w, 'height': h, 'file_name': f'{match_dir}/frame/{rally_id}/{i}.{IMG_FORMAT}'})
            if vis > 0:
                annotations.append({'id': sample_count,
                                    'image_id': sample_count,
                                    'category_id': 1,
                                    'bbox': [int(cx-bbox_size/2), int(cy-bbox_size/2), bbox_size, bbox_size],
                                    'ignore': 0,
                                    'area': bbox_size*bbox_size,
                                    'segmentation': [],
                                    'iscrowd': 0})
            sample_count += 1


    coco_data = {
        'info': {},
        'licenses': [],
        'categories': [{'id': 1, 'name': 'shuttlecock'}],
        'images': image_info,
        'annotations': annotations,
    }
    with open(f'{data_dir}/coco_format_gt.json', 'w') as f:
        json.dump(coco_data, f)
    
################################ Preprocessing Functions ################################
def generate_data_frames(video_file):
    """ Sample frames from the videos in the dataset.

        Args:
            video_file (str): File path of video in dataset
                Format: '{data_dir}/{split}/match{match_id}/video/{rally_id}.mp4'
        
        Returns:
            None
        
        Actions:
            Generate frames from the video and save as image files to the corresponding frame directory
    """

    # Check file format
    try:
        assert video_file[-4:] == '.mp4', 'Invalid video file format.'
    except:
        raise ValueError(f'{video_file} is not a video file.')

    # Check if the video has matched csv file
    file_format_str = os.path.join('{}', 'video', '{}.mp4')
    match_dir, rally_id = parse.parse(file_format_str, video_file)
    csv_file = os.path.join(match_dir, 'csv', f'{rally_id}_ball.csv')
    label_df = pd.read_csv(csv_file, encoding='utf8')
    assert os.path.exists(video_file) and os.path.exists(csv_file), 'Video file or csv file does not exist.'

    rally_dir = os.path.join(match_dir, 'frame', rally_id)
    if not os.path.exists(rally_dir):
        # Haven't processed yet
        os.makedirs(rally_dir)
    else:
        label_df = pd.read_csv(csv_file, encoding='utf8')
        if len(list_dirs(rally_dir)) < len(label_df):
            # Some error has occured, remove the directory and process again
            shutil.rmtree(rally_dir)
            os.makedirs(rally_dir)
        else:
            # Already processed.
            return

    cap = cv2.VideoCapture(video_file)
    frames = []
    success = True

    # Sample frames until video end or exceed the number of labels
    while success and len(frames) != len(label_df):
        success, frame = cap.read()
        if success:
            frames.append(frame)
            cv2.imwrite(os.path.join(rally_dir, f'{len(frames)-1}.{IMG_FORMAT}'), frame)
    
    # Calculate the median of all frames
    median = np.median(np.array(frames), 0)
    median = median[..., ::-1] # BGR to RGB
    np.savez(os.path.join(rally_dir, 'median.npz'), median=median) # Must be lossless, do not save as image format

def get_match_median(match_dir):
    """ Generate and save the match median frame to the corresponding match directory.

        Args:
            match_dir (str): File path of match directory
                Format: '{data_dir}/{split}/match{match_id}'
            
        Returns:
            None
    """

    medians = []

    # For each rally in the match
    rally_dirs = list_dirs(os.path.join(match_dir, 'frame'))
    for rally_dir in rally_dirs:
        file_format_str = os.path.join('{}', 'frame', '{}')
        _, rally_id = parse.parse(file_format_str, rally_dir)

        # Load rally median, if not exist, generate it
        if not os.path.exists(os.path.join(rally_dir, 'median.npz')):
            get_rally_median(os.path.join(match_dir, 'video', f'{rally_id}.mp4'))
        frame = np.load(os.path.join(rally_dir, 'median.npz'))['median']
        medians.append(frame)
    
    # Calculate the median of all rally medians
    median = np.median(np.array(medians), 0)
    np.savez(os.path.join(match_dir, 'median.npz'), median=median) # Must be lossless, do not save as image format

def get_rally_median(video_file):
    """ Generate and save the rally median frame to the corresponding rally directory.

        Args:
            video_file (str): File path of video file
                Format: '{data_dir}/{split}/match{match_id}/video/{rally_id}.mp4'
        
        Returns:
            None
    """
    
    frames = []

    # Get corresponding rally directory
    file_format_str = os.path.join('{}', 'video', '{}.mp4')
    match_dir, rally_id = parse.parse(file_format_str, video_file)
    save_dir = os.path.join(match_dir, 'frame', rally_id)
    
    # Sample frames from the video
    cap = cv2.VideoCapture(video_file)
    success = True
    while success:
        success, frame = cap.read()
        if success:
            frames.append(frame)
    
    # Calculate the median of all frames
    median = np.median(np.array(frames), 0)[..., ::-1] # BGR to RGB
    np.savez(os.path.join(save_dir, 'median.npz'), median=median) # Must be lossless, do not save as image format

def re_generate_median_files(data_dir):
    for split in ['train', 'val', 'test']:
        match_dirs = list_dirs(os.path.join(data_dir, split))
        for match_dir in match_dirs:
            match_name = match_dir.split('/')[-1]
            video_files = list_dirs(os.path.join(match_dir, 'video'))
            for video_file in video_files:
                print(f'Processing {video_file}...')
                get_rally_median(video_file)
            get_match_median(match_dir)
            print(f'Finish processing {match_name}.')