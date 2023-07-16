import os
import cv2
import parse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence

from utils.general import *

def write_to_tb(model_type, tb_writer, losses, val_res, epoch):
    tb_writer.add_scalars(f"{model_type}/WBCE_Loss", {'train': losses[0],
                                                    'val_eval': losses[1]}, epoch)
    if model_type == 'TrackNet':
        tb_writer.add_scalar(f"{model_type}/Accurcy", val_res['accuracy'], epoch)
        tb_writer.add_scalar(f"{model_type}/Precision", val_res['precision'], epoch)
        tb_writer.add_scalar(f"{model_type}/Recall", val_res['recall'], epoch)
        tb_writer.add_scalar(f"{model_type}/F1-score", val_res['f1'], epoch)
        tb_writer.add_scalar(f"{model_type}/Miss_Rate", val_res['miss_rate'], epoch)
        tb_writer.add_scalar(f"{model_type}_Confusion_Metrix/TP", val_res['TP'], epoch)
        tb_writer.add_scalar(f"{model_type}_Confusion_Metrix/TN", val_res['TN'], epoch)
        tb_writer.add_scalar(f"{model_type}_Confusion_Metrix/FP1", val_res['FP1'], epoch)
        tb_writer.add_scalar(f"{model_type}_Confusion_Metrix/FP2", val_res['FP2'], epoch)
        tb_writer.add_scalar(f"{model_type}_Confusion_Metrix/FN", val_res['FN'], epoch)
    else:
        tb_writer.add_scalars(f"{model_type}/Accurcy", {'val_refine': val_res['inpaint']['accuracy'],
                                                        'val_reconstruct': val_res['reconstruct']['accuracy'],
                                                        'val_baseline': val_res['baseline']['accuracy']}, epoch)
        tb_writer.add_scalars(f"{model_type}/Precision", {'val_refine': val_res['inpaint']['precision'],
                                                          'val_reconstruct': val_res['reconstruct']['precision'],
                                                          'val_baseline': val_res['baseline']['precision']}, epoch)
        tb_writer.add_scalars(f"{model_type}/Recall", {'val_refine': val_res['inpaint']['recall'],
                                                       'val_reconstruct': val_res['reconstruct']['recall'],
                                                       'val_baseline': val_res['baseline']['recall']}, epoch)
        tb_writer.add_scalars(f"{model_type}/F1-score", {'val_refine': val_res['inpaint']['f1'],
                                                         'val_reconstruct': val_res['reconstruct']['f1'],
                                                         'val_baseline': val_res['baseline']['f1']}, epoch)
        tb_writer.add_scalars(f"{model_type}/Miss_Rate", {'val_refine': val_res['inpaint']['miss_rate'],
                                                          'val_reconstruct': val_res['reconstruct']['miss_rate'],
                                                          'val_baseline': val_res['baseline']['miss_rate']}, epoch)
        tb_writer.add_scalars(f"{model_type}_Confusion_Metrix/TP", {'val_refine': val_res['inpaint']['TP'],
                                                                    'val_reconstruct': val_res['reconstruct']['TP'],
                                                                    'val_baseline': val_res['baseline']['TP']}, epoch)
        tb_writer.add_scalars(f"{model_type}_Confusion_Metrix/TN", {'val_refine': val_res['inpaint']['TN'],
                                                                    'val_reconstruct': val_res['reconstruct']['TN'],
                                                                    'val_baseline': val_res['baseline']['TN']}, epoch)
        tb_writer.add_scalars(f"{model_type}_Confusion_Metrix/FP1", {'val_refine': val_res['inpaint']['FP1'],
                                                                     'val_reconstruct': val_res['reconstruct']['FP1'],
                                                                     'val_baseline': val_res['baseline']['FP1']}, epoch)
        tb_writer.add_scalars(f"{model_type}_Confusion_Metrix/FP2", {'val_refine': val_res['inpaint']['FP2'],
                                                                     'val_reconstruct': val_res['reconstruct']['FP2'],
                                                                     'val_baseline': val_res['baseline']['FP2']}, epoch)
        tb_writer.add_scalars(f"{model_type}_Confusion_Metrix/FN", {'val_refine': val_res['inpaint']['FN'],
                                                                    'val_reconstruct': val_res['reconstruct']['FN'],
                                                                    'val_baseline': val_res['baseline']['FN']}, epoch)
    tb_writer.flush()

def plot_median_files(data_dir):
    rally_dirs = []
    if not os.path.exists(os.path.join(data_dir, 'median')):
        os.makedirs(os.path.join(data_dir, 'median'))
    
    for split in ['train', 'test', 'val']:
        match_dirs = list_dirs(os.path.join(data_dir, split))
        for match_dir in match_dirs:
            # Write match median frame
            _, match_id = parse.parse('{}/match{}', match_dir)
            if os.path.exists(os.path.join(data_dir, split, f'match{match_id}', 'median.npz')):
                median = np.load(os.path.join(data_dir, split, f'match{match_id}', 'median.npz'))['median'][..., ::-1]
                cv2.imwrite(os.path.join(data_dir, 'median', f'{split}_m{match_id}.png'), median)
            rally_dirs = list_dirs(os.path.join(match_dir, 'frame'))
            for rally_dir in rally_dirs:
                # Write rally median frame
                _, rally_id = parse.parse('{}/frame/{}', rally_dir)
                if os.path.exists(os.path.join(rally_dir, 'median.npz')):
                    median = np.load(os.path.join(rally_dir, 'median.npz'))['median'][..., ::-1]
                    cv2.imwrite(os.path.join(data_dir, 'median', f'{split}_m{match_id}_r{rally_id}.png'), median)

def plot_heatmap_pred_sample(x, y, y_pred, c, save_dir='', bg_mode=''):
    """ Visualize the inupt sequence with its predicted heatmap.
        Save as a gif image.

        args:
            x - A numpy.ndarray of input sequences with shape (L, H, W, 3)
            y - A numpy.ndarray of ground-truth heatmap sequences with shape (L, H, W)
            y_pred - A numpy.ndarray of predicted heatmap sequences with shape (L, H, W)
            c - A numpy.ndarray of ground-truth coordinate sequences with shape (L, 2)
            save_dir - A str specifying the save directory
    """
    imgs = []
    y_map = y_pred > 0.5

    # Scale value from [0, 1] to [0, 255]
    x = to_img(x)
    y = to_img(y)
    y_p = to_img(y_pred)
    y_m = to_img(y_map)
    
    # Write image sequence to gif
    for f in range(c.shape[0]):
        # Stack channels to form RGB images
        tmp_y = cv2.cvtColor(y[f], cv2.COLOR_GRAY2BGR)
        tmp_pred = cv2.cvtColor(y_p[f], cv2.COLOR_GRAY2BGR)
        tmp_map = cv2.cvtColor(y_m[f], cv2.COLOR_GRAY2BGR)
        tmp_x = cv2.cvtColor(x[f], cv2.COLOR_GRAY2BGR) if bg_mode == 'subtract' else x[f]
        assert tmp_x.shape == tmp_y.shape == tmp_pred.shape == tmp_map.shape

        # Mark ground-truth label
        cv2.circle(tmp_x, (int(c[f][0] * WIDTH), int(c[f][1] * HEIGHT)), 2, (255, 0, 0), -1)
        up_img = cv2.hconcat([tmp_x, tmp_y])
        down_img = cv2.hconcat([tmp_pred, tmp_map])
        img = cv2.vconcat([up_img, down_img])

        # Cast cv image to PIL image for saving gif format
        img = Image.fromarray(img)
        imgs.append(img)
        imgs[0].save(f'{save_dir}/pred_cur.gif', format='GIF', save_all=True, append_images=imgs[1:], duration=1000, loop=0)

def plot_traj_pred_sample(gt_coor, refine_coor, inpaint_mask, save_dir=''):
    """ Visualize the inupt sequence with its predicted heatmap.
        Save as a gif image.

        args:
            gt_coor - A numpy.ndarray of ground-truth coordinate sequences with shape (L, 2)
            refine_coor - A numpy.ndarray of predicted coordinate sequences with shape (L, 2)
            inpaint_mask - A numpy.ndarray of inpaint mask sequences with shape (L, 1)
            save_dir - A str specifying the save directory
    """
    img = np.zeros((HEIGHT, WIDTH, 3), dtype='uint8')
    for f in range(gt_coor.shape[0]):
        # Plot ground-truth and predicted trajectory
        img = cv2.circle(img, (int(gt_coor[f][0] * WIDTH), int(gt_coor[f][1] * HEIGHT)), 2, (0, 0, 255), -1)
        if inpaint_mask[f] == 1:
            img = cv2.circle(img, (int(refine_coor[f][0] * WIDTH), int(refine_coor[f][1] * HEIGHT)), 2, (0, 255, 0), -1)
    
    cv2.imwrite(f'{save_dir}/traj_pred.png', img)