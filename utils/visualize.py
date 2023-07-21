import os
import cv2
import parse
import numpy as np
from PIL import Image
from utils.general import *

def write_to_tb(model_type, tb_writer, losses, val_res, epoch):
    """ Write training and validation results to tensorboard. 

        Args:
            model_type (str): Model type
                Choices:'TrackNet', 'InpaintNet'
            tb_writer (tensorboard.SummaryWriter): Tensorboard writer
            losses (Tuple[float, float]): Training and validation losses
            val_res (dict):Validation results
            epoch (int): Current epoch

        Returns:
            None
    """

    if model_type == 'TrackNet':
        tb_writer.add_scalars(f"{model_type}_Loss/WBCE", {'train': losses[0],
                                                          'val': losses[1]}, epoch)
        tb_writer.add_scalar(f"{model_type}_Metric/Accurcy", val_res['accuracy'], epoch)
        tb_writer.add_scalar(f"{model_type}_Metric/Precision", val_res['precision'], epoch)
        tb_writer.add_scalar(f"{model_type}_Metric/Recall", val_res['recall'], epoch)
        tb_writer.add_scalar(f"{model_type}_Metric/F1-score", val_res['f1'], epoch)
        tb_writer.add_scalar(f"{model_type}_Metric/Miss_Rate", val_res['miss_rate'], epoch)
        tb_writer.add_scalar(f"{model_type}_Results/TP", val_res['TP'], epoch)
        tb_writer.add_scalar(f"{model_type}_Results/TN", val_res['TN'], epoch)
        tb_writer.add_scalar(f"{model_type}_Results/FP1", val_res['FP1'], epoch)
        tb_writer.add_scalar(f"{model_type}_Results/FP2", val_res['FP2'], epoch)
        tb_writer.add_scalar(f"{model_type}_Results/FN", val_res['FN'], epoch)
    else:
        tb_writer.add_scalars(f"{model_type}_Loss/MSE", {'train': losses[0],
                                                         'val': losses[1]}, epoch)
        tb_writer.add_scalars(f"{model_type}_Metric/Accurcy", {'val_refine': val_res['inpaint']['accuracy'],
                                                        'val_reconstruct': val_res['reconstruct']['accuracy'],
                                                        'val_baseline': val_res['baseline']['accuracy']}, epoch)
        tb_writer.add_scalars(f"{model_type}_Metric/Precision", {'val_refine': val_res['inpaint']['precision'],
                                                          'val_reconstruct': val_res['reconstruct']['precision'],
                                                          'val_baseline': val_res['baseline']['precision']}, epoch)
        tb_writer.add_scalars(f"{model_type}_Metric/Recall", {'val_refine': val_res['inpaint']['recall'],
                                                       'val_reconstruct': val_res['reconstruct']['recall'],
                                                       'val_baseline': val_res['baseline']['recall']}, epoch)
        tb_writer.add_scalars(f"{model_type}_Metric/F1-score", {'val_refine': val_res['inpaint']['f1'],
                                                         'val_reconstruct': val_res['reconstruct']['f1'],
                                                         'val_baseline': val_res['baseline']['f1']}, epoch)
        tb_writer.add_scalars(f"{model_type}_Metric/Miss_Rate", {'val_refine': val_res['inpaint']['miss_rate'],
                                                          'val_reconstruct': val_res['reconstruct']['miss_rate'],
                                                          'val_baseline': val_res['baseline']['miss_rate']}, epoch)
        tb_writer.add_scalars(f"{model_type}_Results/TP", {'val_refine': val_res['inpaint']['TP'],
                                                                    'val_reconstruct': val_res['reconstruct']['TP'],
                                                                    'val_baseline': val_res['baseline']['TP']}, epoch)
        tb_writer.add_scalars(f"{model_type}_Results/TN", {'val_refine': val_res['inpaint']['TN'],
                                                                    'val_reconstruct': val_res['reconstruct']['TN'],
                                                                    'val_baseline': val_res['baseline']['TN']}, epoch)
        tb_writer.add_scalars(f"{model_type}_Results/FP1", {'val_refine': val_res['inpaint']['FP1'],
                                                                     'val_reconstruct': val_res['reconstruct']['FP1'],
                                                                     'val_baseline': val_res['baseline']['FP1']}, epoch)
        tb_writer.add_scalars(f"{model_type}_Results/FP2", {'val_refine': val_res['inpaint']['FP2'],
                                                                     'val_reconstruct': val_res['reconstruct']['FP2'],
                                                                     'val_baseline': val_res['baseline']['FP2']}, epoch)
        tb_writer.add_scalars(f"{model_type}_Results/FN", {'val_refine': val_res['inpaint']['FN'],
                                                                    'val_reconstruct': val_res['reconstruct']['FN'],
                                                                    'val_baseline': val_res['baseline']['FN']}, epoch)
    tb_writer.flush()

def plot_median_files(data_dir):
    """ Plot median frames of each match and rally and save to '{data_dir}/median'. 
    
        Args:
            data_dir (str): Data root directory
    """

    rally_dirs = []
    if not os.path.exists(os.path.join(data_dir, 'median')):
        os.makedirs(os.path.join(data_dir, 'median'))
    
    for split in ['train', 'test', 'val']:
        match_dirs = list_dirs(os.path.join(data_dir, split))
        # For each match
        for match_dir in match_dirs:
            file_format_str = os.path.join('{}', 'match{}')
            _, match_id = parse.parse(file_format_str, match_dir)
            if os.path.exists(os.path.join(data_dir, split, f'match{match_id}', 'median.npz')):
                median = np.load(os.path.join(data_dir, split, f'match{match_id}', 'median.npz'))['median'][..., ::-1] # BGR to RGB
                cv2.imwrite(os.path.join(data_dir, 'median', f'{split}_m{match_id}.png'), median)
            rally_dirs = list_dirs(os.path.join(match_dir, 'frame'))
            # For each rally
            for rally_dir in rally_dirs:
                file_format_str = os.path.join('{}', 'frame', '{}')
                _, rally_id = parse.parse(file_format_str, rally_dir)
                if os.path.exists(os.path.join(rally_dir, 'median.npz')):
                    median = np.load(os.path.join(rally_dir, 'median.npz'))['median'][..., ::-1] # BGR to RGB
                    cv2.imwrite(os.path.join(data_dir, 'median', f'{split}_m{match_id}_r{rally_id}.png'), median)

def plot_heatmap_pred_sample(x, y, y_pred, c, bg_mode, save_dir):
    """ Visualize input and output of TrackNet and save as a gif. Including 4 subplots:
            Top left: Frames sequence with ball coordinate marked
            Top right: Ground-truth heatmap sequence
            Bottom left: Predicted heatmap sequence
            Bottom right: Predicted heatmap sequence with thresholding

        Args:
            x (numpy.ndarray): Frame sequence with shape (L, H, W)
            y (numpy.ndarray): Ground-truth heatmap sequence with shape (L, H, W)
            y_pred (numpy.ndarray): Predicted heatmap sequence with shape (L, H, W)
            c (numpy.ndarray): Ground-truth ball coordinate sequence with shape (L, 2)
            bg_mode (str): Background mode of TrackNet
            save_dir (str): Save directory
        
        Returns:
            None
    """

    imgs = []

    # Thresholding
    y_map = y_pred > 0.5

    # Convert input and output to image format
    x = to_img(x)
    y = to_img(y)
    y_p = to_img(y_pred)
    y_m = to_img(y_map)
    
    # Write image sequence to gif
    for f in range(c.shape[0]):
        # Convert grayscale image to BGR image for concatenation
        tmp_x = cv2.cvtColor(x[f], cv2.COLOR_GRAY2BGR) if bg_mode == 'subtract' else x[f]
        tmp_y = cv2.cvtColor(y[f], cv2.COLOR_GRAY2BGR)
        tmp_pred = cv2.cvtColor(y_p[f], cv2.COLOR_GRAY2BGR)
        tmp_map = cv2.cvtColor(y_m[f], cv2.COLOR_GRAY2BGR)
        assert tmp_x.shape == tmp_y.shape == tmp_pred.shape == tmp_map.shape

        # Mark ground-truth label
        cv2.circle(tmp_x, (int(c[f][0] * WIDTH), int(c[f][1] * HEIGHT)), 2, (255, 0, 0), -1)

        # Concatenate 4 subplots
        up_img = cv2.hconcat([tmp_x, tmp_y])
        down_img = cv2.hconcat([tmp_pred, tmp_map])
        img = cv2.vconcat([up_img, down_img])

        # Cast cv image to PIL image for saving gif format
        img = Image.fromarray(img)
        imgs.append(img)
        imgs[0].save(f'{save_dir}/cur_pred_TrackNet.gif', format='GIF', save_all=True, append_images=imgs[1:], duration=1000, loop=0)

def plot_traj_pred_sample(coor_gt, coor_inpaint, inpaint_mask, save_dir=''):
    """ Visualize input and output of InpaintNet and save as a png.

        Args:
            coor_gt (numpy.ndarray): Ground-truth trajectory with shape (L, 2)
            coor_inpaint (numpy.ndarray): Inpainted trajectory with shape (L, 2)
            inpaint_mask (numpy.ndarray): Inpainting mask with shape (L, 1)
            save_dir (str): Save directory
        
        Returns:
            None
    """

    # Create an empty image
    img = np.ones((HEIGHT, WIDTH, 3), dtype='uint8')

    # Mark ground-truth and predicted coordinate in the trajectory
    for f in range(coor_gt.shape[0]):
        img = cv2.circle(img, (int(coor_gt[f][0] * WIDTH), int(coor_gt[f][1] * HEIGHT)), 2, (0, 0, 255), -1)
        if inpaint_mask[f] == 1:
            img = cv2.circle(img, (int(coor_inpaint[f][0] * WIDTH), int(coor_inpaint[f][1] * HEIGHT)), 2, (0, 255, 0), -1)
    
    cv2.imwrite(f'{save_dir}/cur_pred_InpaintNet.png', img)