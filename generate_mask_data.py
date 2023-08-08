import os
import torch
import argparse

from test import test
from utils.general import get_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracknet_file', type=str, help='TrackNet model file')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--eval_mode', type=str, default='weight', choices=['nonoverlap', 'average', 'weight'], help='evaluation mode')
    parser.add_argument('--split_list', type=lambda splits: [s for s in splits.split(',')], default='train,val,test', help='split list')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    # Load model
    ckpt = torch.load(args.tracknet_file)

    # Load and set parameters
    param_dict = ckpt['param_dict']
    param_dict['num_workers'] = args.batch_size
    param_dict['batch_size'] = args.batch_size
    param_dict['eval_mode'] = args.eval_mode
    param_dict['debug'] = args.debug
    param_dict['verbose'] = args.verbose
    param_dict['tracknet_seq_len'] = param_dict['seq_len']
    param_dict['output_bbox'] = False
    param_dict['output_gt'] = True
    tracknet = get_model(param_dict['model_name'], param_dict['seq_len'], param_dict['bg_mode']).cuda()
    tracknet.load_state_dict(ckpt['model'])
    
    for split in args.split_list:
        print(f'Generating predicted trajectories and inpainting masks for {split} set...')
        _ = test((tracknet, None), split, param_dict, save_inpaint_mask=True)