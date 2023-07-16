import os
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Shuttlecock_Trajectory_Dataset, data_dir
from utils.general import *
from utils.metric import *

# CUDA_VISIBLE_DEVICES=5 python3 evaluation.py --batch_size 16 --eval_mode weight --model_file 0525_exp/with_val/Adam_BN_ReLU_noBias/OriginalTrackNetV2_f8_e30_b10/model_best.pt --save_dir 0525_exp/with_val/Adam_BN_ReLU_noBias/OriginalTrackNetV2_f8_e30_b10/eval

pred_types = ['TP', 'TN', 'FP1', 'FP2', 'FN']
pred_types_map = {pred_type: i for i, pred_type in enumerate(pred_types)}
eval_types = ['inpaint', 'reconstruct', 'baseline']

# Only for training evaluation without saving detail results
def eval_tracknet(model, data_loader, param_dict):
    model.eval()
    losses = []
    confusion_matrix = np.zeros(5) # TP, TN, FP1, FP2, FN
    if param_dict['verbose']:
        data_prob = tqdm(data_loader)
    else:
        data_prob = data_loader
    
    for step, (i, x, y, _, _) in enumerate(data_prob):
        x, y = x.float().cuda(), y.float().cuda()
        with torch.no_grad():
            y_pred = model(x)

        loss = WBCELoss(y_pred, y)
        losses.append(loss.item())

        tmp_res = get_confusion_matrix(i, y_true=y, y_pred=y_pred, tolerance=param_dict['tolerance'])
        confusion_matrix += tmp_res
        
        if param_dict['verbose']:
            TP, TN, FP1, FP2, FN = confusion_matrix
            data_prob.set_description(f'Evaluation')
            data_prob.set_postfix(TP=TP, TN=TN, FP1=FP1, FP2=FP2, FN=FN)
    
    # for evaluation in training
    TP, TN, FP1, FP2, FN = confusion_matrix
    accuracy, precision, recall, f1, miss_rate = get_metric(TP, TN, FP1, FP2, FN)
    res_dict = {'TP': TP, 'TN': TN, 'FP1': FP1, 'FP2': FP2, 'FN': FN, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'miss_rate': miss_rate}
    return float(np.mean(losses)), res_dict

def eval_inpaintnet(model, data_loader, param_dict):
    model.eval()
    losses = []
    confusion_matrix = {eval_type: np.zeros(5) for eval_type in eval_types} # TP, TN, FP1, FP2, FN
    if param_dict['verbose']:
        data_prob = tqdm(data_loader)
    else:
        data_prob = data_loader

    for step, (i, coor_pred, coor, _, _, inpaint_mask) in enumerate(data_prob):
        coor_pred, coor, inpaint_mask = coor_pred.float().cuda(), coor.float().cuda(), inpaint_mask.float().cuda()
        
        with torch.no_grad():
            coor_inpaint = model(coor_pred, inpaint_mask)
            coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1-inpaint_mask)
            
            loss = nn.MSELoss()(coor_inpaint * inpaint_mask, coor * inpaint_mask)
            losses.append(loss.item())

            th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH)) | (coor_inpaint[:, :, 0] > 1) | (coor_inpaint[:, :, 1] > 1) | (coor_inpaint[:, :, 0] < 0) | (coor_inpaint[:, :, 1] < 0)
            coor_inpaint[th_mask] = 0.
        
        for eval_type in eval_types:
            if eval_type == 'inpaint':
                tmp_res = get_confusion_matrix(i, c_true=coor, c_pred=coor_inpaint, tolerance=param_dict['tolerance'])
            elif eval_type == 'reconstruct':
                tmp_res = get_confusion_matrix(i, c_true=coor_pred, c_pred=coor_inpaint, tolerance=param_dict['tolerance'])
            elif eval_type == 'baseline':
                tmp_res = get_confusion_matrix(i, c_true=coor, c_pred=coor_pred, tolerance=param_dict['tolerance'])
            else:
                raise ValueError('Invalid eval_type')
            confusion_matrix[eval_type] += tmp_res
        
        if param_dict['verbose']:
            TP, TN, FP1, FP2, FN = confusion_matrix['inpaint']
            data_prob.set_description(f'Evaluation')
            data_prob.set_postfix(TP=TP, TN=TN, FP1=FP1, FP2=FP2, FN=FN)
    
    res_dict = {}
    for eval_type in eval_types:
        TP, TN, FP1, FP2, FN = confusion_matrix[eval_type]
        accuracy, precision, recall, f1, miss_rate = get_metric(TP, TN, FP1, FP2, FN)
        res_dict[eval_type] = {'TP': TP, 'TN': TN, 'FP1': FP1, 'FP2': FP2, 'FN': FN, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'miss_rate': miss_rate}
    return float(np.mean(losses)), res_dict

# For testing evaluation
def gen_inpaint_mask(pred_dict, y_max, y_th_ratio=0.05):
    y = np.array(pred_dict['Y'])
    vis_pred = np.array(pred_dict['Visibility'])
    inpaint_mask = np.zeros_like(y)
    i = 0 # index that ball start to disappear
    j = 0 # index that ball start to appear
    threshold = y_max * y_th_ratio
    while j < len(vis_pred):
        while i < len(vis_pred) and vis_pred[i] == 1:
            i += 1
        j = i
        while j < len(vis_pred) and vis_pred[j] == 0:
            j += 1
        if j == i:
            break
        elif i == 0 and y[j] > threshold:
            # start from the first frame that ball disappear
            inpaint_mask[:j] = 1
        elif (i > 1 and y[i-1] > threshold) and (j < len(vis_pred) and y[j] > threshold):
            inpaint_mask[i:j] = 1
        else:
            # ball is out of the field of camera view 
            pass
        i = j
    
    return inpaint_mask.tolist()

def get_eval_res(pred_dict, drop=False):
    res_dict = {pred_type: 0 for pred_type in pred_types}
    for rally_key, pred in pred_dict.items():
        if drop:
            drop_frame_dict = json.load(open(f'{data_dir}/drop_frame.json'))
            start_f, end_f = drop_frame_dict['start'], drop_frame_dict['end']
            type_res = np.array(pred['Type'])[start_f[rally_key]:end_f[rally_key]]
        else:
            type_res = np.array(pred['Type'])
        
        # Calculate metrics
        for pred_type in pred_types:
            res_dict[pred_type] += int((type_res == pred_types_map[pred_type]).sum())
    
    TP, TN, FP1, FP2, FN = res_dict['TP'], res_dict['TN'], res_dict['FP1'], res_dict['FP2'], res_dict['FN']
    accuracy, precision, recall, f1, miss_rate = get_metric(TP, TN, FP1, FP2, FN)
    res_dict = {'TP': TP, 'TN': TN, 'FP1': FP1, 'FP2': FP2, 'FN': FN,
                'accuracy': accuracy, 'precision': precision, 'recall': recall,
                'f1': f1, 'miss_rate': miss_rate}
    return res_dict

def get_ensemble_weight(seq_len, eval_mode):
    if eval_mode == 'average':
        weight = torch.ones(seq_len) / seq_len
    elif eval_mode == 'weight':
        weight = torch.ones(seq_len)
        for i in range(math.ceil(seq_len/2)):
            weight[i] = (i+1)
            weight[seq_len-i-1] = (i+1)
        weight = weight / weight.sum()
    else:
        raise ValueError('Invalid mode')
    return weight

def perdict(indices, y_true=None, y_pred=None, c_true=None, c_pred=None, tolerance=4, img_scaler=1):
    """
        Predict and output the result of each frame
        args:
            indices - A tf.EagerTensor or a list of indices for sequences with shape (N)
            y_true - A tf.EagerTensor of ground-truth heatmap sequences
            y_pred - A tf.EagerTensor of predicted heatmap sequences
            c_true - A tf.EagerTensor of ground-truth coordinate sequences with shape (N, 2)
            c_pred - A tf.EagerTensor of predicted coordinate sequences with shape (N, 2)
            tolerance - A int speicfying the tolerance for FP1
            img_scaler - A float specifying the scaler for image size
        
        returns:
            pred_dict - A dict of prediction results
            detection_list - A list of detection results
    """
    
    pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Type':[]}
    #detection_list = []
    """ Detection Format
        [{
            "image_id": int,
            "category_id": int,
            "bbox": [x,y,width,height],
            "score": float,
        }]
    """

    indices = indices.detach().cpu().numpy().tolist() if torch.is_tensor(indices) else indices.numpy().tolist()

    if y_true is not None and y_pred is not None:
        y_pred = y_pred > 0.5
        batch_size, seq_len = y_true.shape[0], y_true.shape[1]
        y_true = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else y_true
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_true = to_img_format(y_true) # (N, F, H, W)
        y_pred = to_img_format(y_pred) # (N, F, H, W)
    
    if c_true is not None and c_pred is not None:
        batch_size, seq_len = c_true.shape[0], c_true.shape[1]
        c_true = c_true.detach().cpu().numpy() if torch.is_tensor(c_true) else c_true
        c_pred = c_pred.detach().cpu().numpy() if torch.is_tensor(c_pred) else c_pred
        c_true[:, :, 0] = c_true[:, :, 0] * WIDTH
        c_true[:, :, 1] = c_true[:, :, 1] * HEIGHT
        c_pred[:, :, 0] = c_pred[:, :, 0] * WIDTH
        c_pred[:, :, 1] = c_pred[:, :, 1] * HEIGHT

    prev_f_i = -1
    for n in range(batch_size):
        for f in range(seq_len):
            f_i = int(indices[n][f][1])
            if f_i != prev_f_i:
                if c_true is not None and c_pred is not None:
                    c_t = c_true[n][f]
                    c_p = c_pred[n][f]
                    cx_pred, cy_pred = int(c_p[0]), int(c_p[1])
                    vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                    if np.amax(c_p) == 0 and np.amax(c_t) == 0:
                        # True Negative: prediction is no ball, and ground truth is no ball
                        pred_dict['Type'].append(pred_types_map['TN'])
                    elif np.amax(c_p) > 0 and np.amax(c_t) == 0:
                        # False Positive 2: prediction is ball existing, but ground truth is no ball
                        pred_dict['Type'].append(pred_types_map['FP2'])
                    elif np.amax(c_p) == 0 and np.amax(c_t) > 0:
                        # False Negative: prediction is no ball, but ground truth is ball existing
                        pred_dict['Type'].append(pred_types_map['FN'])
                    elif np.amax(c_p) > 0 and np.amax(c_t) > 0:
                        # Both prediction and ground truth are ball existing
                        cx_true, cy_true = int(c_t[0]), int(c_t[1])
                        dist = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
                        if dist > tolerance:
                            # False Positive 1: prediction is ball existing, but is too far from ground truth
                            pred_dict['Type'].append(pred_types_map['FP1'])
                        else:
                            # True Positive
                            pred_dict['Type'].append(pred_types_map['TP'])
                    else:
                        raise ValueError(f'Invalid input: {c_p}, {c_t}')
                elif y_true is not None and y_pred is not None:
                    y_t = y_true[n][f]
                    y_p = y_pred[n][f]
                    cx_pred, cy_pred = predict_location(to_img(y_p))
                    vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                    if np.amax(y_p) == 0 and np.amax(y_t) == 0:
                        # True Negative: prediction is no ball, and ground truth is no ball
                        pred_dict['Type'].append(pred_types_map['TN'])
                    elif np.amax(y_p) > 0 and np.amax(y_t) == 0:
                        # False Positive 2: prediction is ball existing, but ground truth is no ball
                        pred_dict['Type'].append(pred_types_map['FP2'])
                    elif np.amax(y_p) == 0 and np.amax(y_t) > 0:
                        # False Negative: prediction is no ball, but ground truth is ball existing
                        pred_dict['Type'].append(pred_types_map['FN'])
                    elif np.amax(y_p) > 0 and np.amax(y_t) > 0:
                        # Both prediction and ground truth are ball existing
                        # Find center coordinate of the contour with max area as prediction
                        cx_true, cy_true = predict_location(to_img(y_t))
                        dist = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
                        if dist > tolerance:
                            # False Positive 1: prediction is ball existing, but is too far from ground truth
                            pred_dict['Type'].append(pred_types_map['FP1'])
                        else:
                            # True Positive
                            pred_dict['Type'].append(pred_types_map['TP'])
                    else:
                        raise ValueError('Invalid input')
                else:
                    raise ValueError('Invalid input')
                pred_dict['Frame'].append(f_i)
                pred_dict['X'].append(int(cx_pred*img_scaler))
                pred_dict['Y'].append(int(cy_pred*img_scaler))
                pred_dict['Visibility'].append(vis_pred)
            else:
                break
    
    return pred_dict

def test(model, split, param_dict, save_inpaint_mask=False):
    pred_dict = {}
    rally_dirs = get_rally_dirs(data_dir, split)
    rally_dirs = [os.path.join(data_dir, rally_dir) for rally_dir in rally_dirs]
    for rally_dir in rally_dirs:
        _, match_id, rally_id = parse.parse('{}/match{}/frame/{}', rally_dir)
        rally_key = f'{match_id}_{rally_id}'
        tmp_pred = test_rally(model, rally_dir, param_dict)
        pred_dict[rally_key] = tmp_pred
        if save_inpaint_mask:
            write_pred_csv(rally_dir, tmp_pred, save_inpaint_mask=save_inpaint_mask)
    return pred_dict

def test_rally(model, rally_dir, param_dict):
    tracknet, inpaintnet = model
    tracknet.eval()
    
    seq_len = param_dict['tracknet_seq_len']
    pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Type':[]}
    w, h = Image.open(os.path.join(rally_dir, '0.png')).size
    img_scaler = w / WIDTH

    # Test on TrackNet
    if param_dict['eval_mode'] == 'nonoverlap':
        dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='heatmap', bg_mode=param_dict['bg_mode'], rally_dir=rally_dir, padding=True)
        data_loader = DataLoader(dataset, batch_size=param_dict['batch_size'], shuffle=False, num_workers=param_dict['num_workers'], drop_last=False)
        for step, (i, x, y, _, _) in enumerate(tqdm(data_loader)):
            x = x.float().cuda()
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()
            
            # predict
            tmp_pred = perdict(i, y_true=y, y_pred=y_pred, tolerance=param_dict['tolerance'], img_scaler=img_scaler)
            for key in tmp_pred.keys():
                pred_dict[key].extend(tmp_pred[key])
    else:
        weight = get_ensemble_weight(seq_len, param_dict['eval_mode'])
        dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1, data_mode='heatmap', bg_mode=param_dict['bg_mode'], rally_dir=rally_dir)
        data_loader = DataLoader(dataset, batch_size=param_dict['batch_size'], shuffle=False, num_workers=param_dict['num_workers'], drop_last=False)
        num_batch = len(data_loader)

        # Init buffer which keep previous frame prediction
        buffer_size = seq_len - 1
        batch_i = torch.arange(seq_len) # [0, 1, 2, 3, 4, 5, 6, 7]
        frame_i = torch.arange(seq_len-1, -1, -1) # [7, 6, 5, 4, 3, 2, 1, 0]
        y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
        
        for step, (i, x, y, _, _) in enumerate(tqdm(data_loader)):
            x = x.float().cuda()
            b_size, seq_len = i.shape[0], i.shape[1]
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()
            
            # Temporal ensemble
            y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
            ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
            ensemble_y = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)
            ensemble_y_pred = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)

            if step == num_batch-1:
                # Last batch
                y_zero_pad = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
                y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)
                count = buffer_size
                for b in range(b_size+buffer_size):
                    if b >= b_size:
                        # Last input sequence
                        y_pred = y_pred_buffer[batch_i+b, frame_i].sum(0)
                        y_pred /= count
                        frame_idx = seq_len-count
                        ensemble_i = torch.cat((ensemble_i, i[-1][frame_idx].reshape(1, 1, 2)), dim=0)
                        ensemble_y = torch.cat((ensemble_y, y[-1][frame_idx].reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                        ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                        count -= 1
                    else:
                        y_pred = (y_pred_buffer[batch_i+b, frame_i] * weight[:, None, None]).sum(0)
                        ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
                        ensemble_y = torch.cat((ensemble_y, y[b][0].reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                        ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
            else:
                for b in range(b_size):
                    if step == 0 and b < buffer_size:
                        # First batch
                        y_pred = y_pred_buffer[batch_i+b, frame_i].sum(0)
                        y_pred /= (b+1)
                    else:
                        y_pred = (y_pred_buffer[batch_i+b, frame_i] * weight[:, None, None]).sum(0)
                    
                    ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
                    ensemble_y = torch.cat((ensemble_y, y[b][0].reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                    ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                    
            # Predict
            tmp_pred = perdict(ensemble_i, y_true=ensemble_y, y_pred=ensemble_y_pred,
                               tolerance=param_dict['tolerance'], img_scaler=img_scaler)
            for key in tmp_pred.keys():
                pred_dict[key].extend(tmp_pred[key])

            # keep last predictions for ensemble in next iteration
            y_pred_buffer = y_pred_buffer[-(seq_len-1):]
    
    # Test on TrackNetV3 (TrackNet + InpaintNet)
    if inpaintnet is not None:
        inpaintnet.eval()
        seq_len = param_dict['inpaintnet_seq_len']
        #pred_dict = {eval_type: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Type':[]} for eval_type in eval_types}
        pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Type':[]}

        if param_dict['eval_mode'] == 'nonoverlap':
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='coordinate', rally_dir=rally_dir, padding=True)
            data_loader = DataLoader(dataset, batch_size=param_dict['batch_size'], shuffle=False, num_workers=param_dict['num_workers'], drop_last=False)
            for step, (i, coor_pred, coor, _, _, inpaint_mask) in enumerate(tqdm(data_loader)):
                coor_pred, coor, inpaint_mask = coor_pred.float(), coor.float(), inpaint_mask.float()
                with torch.no_grad():
                    coor_inpaint = inpaintnet(coor_pred.cuda(), inpaint_mask.cuda()).detach().cpu()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1-inpaint_mask)
                
                th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH)) | (coor_inpaint[:, :, 0] > 1) | (coor_inpaint[:, :, 1] > 1) | (coor_inpaint[:, :, 0] < 0) | (coor_inpaint[:, :, 1] < 0)
                coor_inpaint[th_mask] = 0.
                
                # predict
                
                tmp_pred = perdict(i, c_true=coor, c_pred=coor_inpaint, tolerance=param_dict['tolerance'], img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    pred_dict[key].extend(tmp_pred[key])
                '''for eval_type in eval_types:
                    if eval_type == 'inpaint':
                        tmp_pred = perdict(i, c_true=coor, c_pred=coor_inpaint, tolerance=param_dict['tolerance'], img_scaler=img_scaler)
                    elif eval_type == 'reconstruct':
                        tmp_pred = perdict(i, c_true=coor_pred, c_pred=coor_inpaint, tolerance=param_dict['tolerance'], img_scaler=img_scaler)
                    elif eval_type == 'baseline':
                        tmp_pred = perdict(i, c_true=coor, c_pred=coor_pred, tolerance=param_dict['tolerance'], img_scaler=img_scaler)
                    else:
                        raise ValueError('Invalid eval_type')
                    for key in tmp_pred.keys():
                        pred_dict[eval_type][key].extend(tmp_pred[key])'''
        else:
            weight = get_ensemble_weight(seq_len, param_dict['eval_mode'])
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1, data_mode='coordinate', rally_dir=rally_dir)
            data_loader = DataLoader(dataset, batch_size=param_dict['batch_size'], shuffle=False, num_workers=param_dict['num_workers'], drop_last=False)
            num_batch = len(data_loader)

            buffer_size = seq_len - 1
            batch_i = torch.arange(seq_len) # [0, 1, 2, 3, 4, 5, 6, 7]
            frame_i = torch.arange(seq_len-1, -1, -1) # [7, 6, 5, 4, 3, 2, 1, 0]
            coor_inpaint_buffer = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
            
            for step, (i, coor_pred, coor, _, _, inpaint_mask) in enumerate(tqdm(data_loader)):
                coor_pred, coor, inpaint_mask = coor_pred.float(), coor.float(), inpaint_mask.float()
                b_size = i.shape[0]
                with torch.no_grad():
                    coor_inpaint = inpaintnet(coor_pred.cuda(), inpaint_mask.cuda()).detach().cpu()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1-inpaint_mask)
                
                th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH)) | (coor_inpaint[:, :, 0] > 1) | (coor_inpaint[:, :, 1] > 1) | (coor_inpaint[:, :, 0] < 0) | (coor_inpaint[:, :, 1] < 0)
                coor_inpaint[th_mask] = 0.

                # ensemble
                coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_inpaint), dim=0)
                ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
                ensemble_coor_pred = torch.empty((0, 1, 2), dtype=torch.float32)
                ensemble_coor = torch.empty((0, 1, 2), dtype=torch.float32)
                ensemble_coor_inpaint = torch.empty((0, 1, 2), dtype=torch.float32)
                
                if step == num_batch-1:
                    # last batch
                    coor_zero_pad = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
                    coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_zero_pad), dim=0)
                    count = buffer_size
                    for b in range(b_size+buffer_size):
                        if b >= b_size:
                            # last input sequence
                            coor_inpaint = coor_inpaint_buffer[batch_i+b, frame_i].sum(0)
                            coor_inpaint /= count
                            
                            frame_idx = seq_len-count
                            ensemble_i = torch.cat((ensemble_i, i[-1][frame_idx].view(1, 1, 2)), dim=0)
                            ensemble_coor_pred = torch.cat((ensemble_coor_pred, coor_pred[-1][frame_idx].view(1, 1, 2)), dim=0)
                            ensemble_coor = torch.cat((ensemble_coor, coor[-1][frame_idx].view(1, 1, 2)), dim=0)
                            ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)
                            count -= 1
                        else:
                            coor_inpaint = (coor_inpaint_buffer[batch_i+b, frame_i] * weight[:, None]).sum(0)
                            ensemble_i = torch.cat((ensemble_i, i[b][0].view(1, 1, 2)), dim=0)
                            ensemble_coor_pred = torch.cat((ensemble_coor_pred, coor_pred[b][0].view(1, 1, 2)), dim=0)
                            ensemble_coor = torch.cat((ensemble_coor, coor[b][0].view(1, 1, 2)), dim=0)
                            ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)
                else:
                    for b in range(b_size):
                        if step == 0 and b < buffer_size:
                            # first batch
                            coor_inpaint = coor_inpaint_buffer[batch_i+b, frame_i].sum(0)
                            coor_inpaint /= (b+1)
                        else:
                            coor_inpaint = (coor_inpaint_buffer[batch_i+b, frame_i] * weight[:, None]).sum(0)
                        
                        ensemble_i = torch.cat((ensemble_i, i[b][0].view(1, 1, 2)), dim=0)
                        ensemble_coor_pred = torch.cat((ensemble_coor_pred, coor_pred[b][0].view(1, 1, 2)), dim=0)
                        ensemble_coor = torch.cat((ensemble_coor, coor[b][0].view(1, 1, 2)), dim=0)
                        ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)

                th_mask = ((ensemble_coor_inpaint[:, :, 0] < COOR_TH) & (ensemble_coor_inpaint[:, :, 1] < COOR_TH)) | (ensemble_coor_inpaint[:, :, 0] > 1) | (ensemble_coor_inpaint[:, :, 1] > 1) | (ensemble_coor_inpaint[:, :, 0] < 0) | (ensemble_coor_inpaint[:, :, 1] < 0)
                ensemble_coor_inpaint[th_mask] = 0.

                # Predict
                tmp_pred = perdict(ensemble_i, c_true=ensemble_coor, c_pred=ensemble_coor_inpaint,
                                    tolerance=param_dict['tolerance'], img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    pred_dict[key].extend(tmp_pred[key])
                '''for eval_type in eval_types:
                    if eval_type == 'inpaint':
                        tmp_pred = perdict(ensemble_i, c_true=ensemble_coor, c_pred=ensemble_coor_inpaint,
                                           tolerance=param_dict['tolerance'], img_scaler=img_scaler)
                    elif eval_type == 'reconstruct':
                        tmp_pred = perdict(ensemble_i, c_true=ensemble_coor_pred, c_pred=ensemble_coor_inpaint,
                                           tolerance=param_dict['tolerance'], img_scaler=img_scaler)
                    elif eval_type == 'baseline':
                        tmp_pred = perdict(ensemble_i, c_true=ensemble_coor, c_pred=ensemble_coor_pred,
                                           tolerance=param_dict['tolerance'], img_scaler=img_scaler)
                    else:
                        raise ValueError('Invalid eval_type')
                    for key in tmp_pred.keys():
                        pred_dict[eval_type][key].extend(tmp_pred[key])'''

                # keep last predictions for ensemble in next iteration
                coor_inpaint_buffer = coor_inpaint_buffer[-(seq_len-1):]

        return pred_dict
    else:
        pred_dict['Inpainting'] = gen_inpaint_mask(pred_dict, y_max=h, y_th_ratio=0.05)
        return pred_dict

def write_pred_video(rally_dir, pred_dict, save_dir):
    match_dir, rally_id = parse.parse('{}/frame/{}', rally_dir)
    csv_file = os.path.join(match_dir, 'corrected_csv', f'{rally_id}_ball.csv') if 'test' in rally_dir else os.path.join(match_dir, 'csv', f'{rally_id}_ball.csv')
    label_df = pd.read_csv(csv_file, encoding='utf8').sort_values(by='Frame').fillna(0)
    f_i, x, y, vis = label_df['Frame'], label_df['X'], label_df['Y'], label_df['Visibility']
    x_pred, y_pred, vis_pred = pred_dict['X'], pred_dict['Y'], pred_dict['Visibility']

    # Video config
    out_video_file = os.path.join(save_dir, f'{rally_id}.mp4')
    img_file = os.path.join(rally_dir, '0.png')
    w, h = Image.open(img_file).size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_file, fourcc, 30, (w, h))

    # Draw prediction
    for i in f_i:
        img = cv2.imread(os.path.join(rally_dir, f'{i}.png'))
        if vis[i]:
            cv2.circle(img, (x[i], y[i]), 5, (0, 0, 255), -1)
        if vis_pred[i]:
            cv2.circle(img, (x_pred[i], y_pred[i]), 5, (0, 255, 0), -1)
        out.write(img)
    out.release()

def write_pred_csv(rally_dir, pred_dict, save_dir=None, save_inpaint_mask=False):
    match_dir, rally_id = parse.parse('{}/frame/{}', rally_dir)
    if save_dir:
        csv_file = os.path.join(save_dir, f'{rally_id}_ball.csv')
    else:
        if not os.path.exists(os.path.join(match_dir, 'predicted_csv')):
            os.makedirs(os.path.join(match_dir, 'predicted_csv'))
        csv_file = os.path.join(match_dir, 'predicted_csv',f'{rally_id}_ball.csv')
    
    if save_inpaint_mask:
        pred_df = pd.DataFrame({'Frame': pred_dict['Frame'], 'Visibility': pred_dict['Visibility'], 'X': pred_dict['X'], 'Y': pred_dict['Y'], 'Inpainting': pred_dict['Inpainting']})
    else:
        pred_df = pd.DataFrame({'Frame': pred_dict['Frame'], 'Visibility': pred_dict['Visibility'], 'X': pred_dict['X'], 'Y': pred_dict['Y']})
    pred_df.to_csv(csv_file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracknet_file', type=str, help='TrackNet model file')
    parser.add_argument('--inpaintnet_file', type=str, default='', help='InpaintNet model file')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='dataset split')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--tolerance', type=float, default=4, help='tolerance of center distance between prediction and ground truth')
    parser.add_argument('--eval_mode', type=str, default='weight', choices=['nonoverlap', 'average', 'weight'], help='evaluation mode')
    parser.add_argument('--video_file', type=str, default='', help='test on a single video with label')
    parser.add_argument('--output_pred', action='store_true', default=False, help='output prediction detail results')
    parser.add_argument('--save_dir', type=str, default='output', help='output directory')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    param_dict = vars(args)
    param_dict['num_workers'] = 8
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Load parameter
    print(f'Loading checkpoint...')
    if args.tracknet_file:
        tracknet_ckpt = torch.load(args.tracknet_file)
        param_dict['tracknet_seq_len'] = tracknet_ckpt['param_dict']['seq_len']
        param_dict['bg_mode'] = tracknet_ckpt['param_dict']['bg_mode']
        tracknet = get_model('TrackNet', seq_len=param_dict['tracknet_seq_len'], bg_mode=param_dict['bg_mode']).cuda()
        tracknet.load_state_dict(tracknet_ckpt['model'])
        model = (tracknet, None)
    
    if args.inpaintnet_file:
        assert args.tracknet_file
        inpaintnet_ckpt = torch.load(args.inpaintnet_file)
        param_dict['inpaintnet_seq_len'] = inpaintnet_ckpt['param_dict']['seq_len']
        inpaintnet = get_model('InpaintNet').cuda()
        inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
        model = (tracknet, inpaintnet)

    if args.video_file:
        print(f'Test on video {args.video_file} ...')
        match_dir, rally_id = parse.parse('{}/video/{}.mp4', args.video_file)
        rally_dir = os.path.join(match_dir, 'frame', rally_id)
        pred_dict = test_rally(model, rally_dir, param_dict)
        if args.inpaintnet_file:
            write_pred_video(rally_dir, pred_dict['inpaint'], args.save_dir)
            write_pred_csv(rally_dir, pred_dict['inpaint'], args.save_dir)
        else:
            write_pred_video(rally_dir, pred_dict, args.save_dir)
            write_pred_csv(rally_dir, pred_dict, args.save_dir)
    else:
        # Evaluation on dataset
        eval_analysis_file = os.path.join(args.save_dir, f'{args.split}_eval_analysis_{args.eval_mode}.json') # for error analysis interface
        eval_res_file = os.path.join(args.save_dir, f'{args.split}_eval_res_{args.eval_mode}.json')

        if not os.path.exists(eval_res_file):
            start_time = time.time()
            print(f'Split: {args.split}')
            print(f'Evaluation mode: {args.eval_mode}')
            print(f'Tolerance Value: {args.tolerance}')
            
            pred_dict = test(model, args.split, param_dict)
            
            if args.split == 'test':
                # drop samples which is not in effective trajectory
                res_dict = get_eval_res(pred_dict, drop=True)
            else:
                res_dict = get_eval_res(pred_dict, drop=False)
            
            with open(eval_res_file, 'w') as f:
                json.dump(res_dict, f, indent=2)
    
        if args.output_pred:
            eval_dict = dict(param_dict=param_dict, pred_dict=pred_dict)
            with open(eval_analysis_file, 'w') as f:
                json.dump(eval_dict, f, indent=2)
