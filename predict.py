import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from test import predict, get_ensemble_weight, generate_inpaint_mask
from dataset import Shuttlecock_Trajectory_Dataset
from utils.general import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str)
    parser.add_argument('--tracknet_file', type=str)
    parser.add_argument('--inpaintnet_file', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_mode', type=str, default='weight', choices=['average', 'weight'])
    parser.add_argument('--save_dir', type=str, default='pred_result')
    parser.add_argument('--output_video', action='store_true', default=False)
    parser.add_argument('--traj_len', type=int, default=8)
    args = parser.parse_args()

    num_workers = 8
    video_name = args.video_file.split('/')[-1][:-4]
    out_csv_file = os.path.join(args.save_dir, f'{video_name}_ball.csv')
    out_video_file = os.path.join(args.save_dir, f'{video_name}.mp4')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Load model
    tracknet_ckpt = torch.load(args.tracknet_file)
    tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
    bg_mode = tracknet_ckpt['param_dict']['bg_mode']
    tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).cuda()
    tracknet.load_state_dict(tracknet_ckpt['model'])
    tracknet.eval()

    if args.inpaintnet_file:
        inpaintnet_ckpt = torch.load(args.inpaintnet_file)
        inpaintnet_seq_len = inpaintnet_ckpt['param_dict']['seq_len']
        inpaintnet = get_model('InpaintNet').cuda()
        inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
        inpaintnet.eval()
    else:
        inpaintnet = None

    # Sample all frames from video
    frame_list, fps, (w, h) = generate_frames(args.video_file)
    w_scaler, h_scaler = w / WIDTH, h / HEIGHT
    img_scaler = (w_scaler, h_scaler)
    print(f'Number of sampled frames: {len(frame_list)}')

    tracknet_pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Inpaint_Mask':[],
                        'Img_scaler': (w_scaler, h_scaler), 'Img_shape': (w, h)}

    # Test on TrackNet
    seq_len = tracknet_seq_len
    if args.eval_mode == 'nonoverlap':
        # Non-overlap sliding window
        dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='heatmap', bg_mode=bg_mode,
                                                frame_arr=np.array(frame_list)[:, :, :, ::-1], padding=True)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        for step, (i, x) in enumerate(tqdm(data_loader)):
            x = x.float().cuda()
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()
            
            # predict
            tmp_pred = predict(i, y_pred=y_pred, img_scaler=img_scaler)
            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])
    else:
        # Temporal ensemble
        dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1, data_mode='heatmap', bg_mode=bg_mode, frame_arr=np.array(frame_list)[:, :, :, ::-1])
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        num_batch = len(data_loader)
        weight = get_ensemble_weight(seq_len, args.eval_mode)

        # Init buffer which keep previous frame prediction
        buffer_size = seq_len - 1
        batch_i = torch.arange(seq_len) # [0, 1, 2, 3, 4, 5, 6, 7]
        frame_i = torch.arange(seq_len-1, -1, -1) # [7, 6, 5, 4, 3, 2, 1, 0]
        y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
        
        for step, (i, x) in enumerate(tqdm(data_loader)):
            x = x.float().cuda()
            b_size, seq_len = i.shape[0], i.shape[1]
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()
            
            y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
            ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
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
                        ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                        count -= 1
                    else:
                        y_pred = (y_pred_buffer[batch_i+b, frame_i] * weight[:, None, None]).sum(0)
                        ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
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
                    ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                    
            # Predict
            tmp_pred = predict(ensemble_i, y_pred=ensemble_y_pred, img_scaler=img_scaler)
            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])

            # Update buffer, keep last predictions for ensemble in next iteration
            y_pred_buffer = y_pred_buffer[-(seq_len-1):]

    # Test on TrackNetV3 (TrackNet + InpaintNet)
    if inpaintnet is not None:
        inpaintnet.eval()
        seq_len = inpaintnet_seq_len
        tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(tracknet_pred_dict, y_max=h, y_th_ratio=0.05)
        inpaint_pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

        if args.eval_mode == 'nonoverlap':
            # Non-overlap sliding window
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='coordinate', pred_dict=tracknet_pred_dict, padding=True)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
            for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader)):
                coor_pred, coor, inpaint_mask = coor_pred.float(), coor.float(), inpaint_mask.float()
                with torch.no_grad():
                    coor_inpaint = inpaintnet(coor_pred.cuda(), inpaint_mask.cuda()).detach().cpu()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1-inpaint_mask)
                
                th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH)) | (coor_inpaint[:, :, 0] > 1) | (coor_inpaint[:, :, 1] > 1) | (coor_inpaint[:, :, 0] < 0) | (coor_inpaint[:, :, 1] < 0)
                coor_inpaint[th_mask] = 0.
                
                # predict
                tmp_pred = predict(i, c_pred=coor_inpaint, img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    inpaint_pred_dict[key].extend(tmp_pred[key])
                
        else:
            # Temporal ensemble
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1, data_mode='coordinate', pred_dict=tracknet_pred_dict)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
            num_batch = len(data_loader)
            weight = get_ensemble_weight(seq_len, args.eval_mode)

            # Init buffer which keep previous frame prediction
            buffer_size = seq_len - 1
            batch_i = torch.arange(seq_len) # [0, 1, 2, 3, 4, 5, 6, 7]
            frame_i = torch.arange(seq_len-1, -1, -1) # [7, 6, 5, 4, 3, 2, 1, 0]
            coor_inpaint_buffer = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
            
            for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader)):
                coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
                b_size = i.shape[0]
                with torch.no_grad():
                    coor_inpaint = inpaintnet(coor_pred.cuda(), inpaint_mask.cuda()).detach().cpu()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1-inpaint_mask)
                
                th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH)) | (coor_inpaint[:, :, 0] > 1) | (coor_inpaint[:, :, 1] > 1) | (coor_inpaint[:, :, 0] < 0) | (coor_inpaint[:, :, 1] < 0)
                coor_inpaint[th_mask] = 0.

                coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_inpaint), dim=0)
                ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
                ensemble_coor_inpaint = torch.empty((0, 1, 2), dtype=torch.float32)
                
                if step == num_batch-1:
                    # Last batch
                    coor_zero_pad = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
                    coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_zero_pad), dim=0)
                    count = buffer_size
                    for b in range(b_size+buffer_size):
                        if b >= b_size:
                            # Last input sequence
                            coor_inpaint = coor_inpaint_buffer[batch_i+b, frame_i].sum(0)
                            coor_inpaint /= count
                            
                            frame_idx = seq_len-count
                            ensemble_i = torch.cat((ensemble_i, i[-1][frame_idx].view(1, 1, 2)), dim=0)
                            ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)
                            count -= 1
                        else:
                            coor_inpaint = (coor_inpaint_buffer[batch_i+b, frame_i] * weight[:, None]).sum(0)
                            ensemble_i = torch.cat((ensemble_i, i[b][0].view(1, 1, 2)), dim=0)
                            ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)
                else:
                    for b in range(b_size):
                        if step == 0 and b < buffer_size:
                            # First batch
                            coor_inpaint = coor_inpaint_buffer[batch_i+b, frame_i].sum(0)
                            coor_inpaint /= (b+1)
                        else:
                            coor_inpaint = (coor_inpaint_buffer[batch_i+b, frame_i] * weight[:, None]).sum(0)
                        
                        ensemble_i = torch.cat((ensemble_i, i[b][0].view(1, 1, 2)), dim=0)
                        ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)

                th_mask = ((ensemble_coor_inpaint[:, :, 0] < COOR_TH) & (ensemble_coor_inpaint[:, :, 1] < COOR_TH)) | (ensemble_coor_inpaint[:, :, 0] > 1) | (ensemble_coor_inpaint[:, :, 1] > 1) | (ensemble_coor_inpaint[:, :, 0] < 0) | (ensemble_coor_inpaint[:, :, 1] < 0)
                ensemble_coor_inpaint[th_mask] = 0.

                # Predict
                tmp_pred = predict(ensemble_i, c_pred=ensemble_coor_inpaint, img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    inpaint_pred_dict[key].extend(tmp_pred[key])
                
                # Update buffer, keep last predictions for ensemble in next iteration
                coor_inpaint_buffer = coor_inpaint_buffer[-(seq_len-1):]
        

    # Write csv file
    pred_dict = inpaint_pred_dict if inpaintnet is not None else tracknet_pred_dict
    write_pred_csv(pred_dict, save_file=out_csv_file)

    # Write video with predicted coordinates
    if args.output_video:
        write_pred_video(frame_list, dict(fps=fps, shape=(w, h)), pred_dict, save_file=out_video_file, traj_len=args.traj_len)

    print('Done.')