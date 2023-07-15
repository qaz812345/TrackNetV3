import os
import cv2
import math
import parse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.general import get_rally_dirs, get_match_median

data_dir = ''

class Shuttlecock_Trajectory_Dataset(Dataset):
    def __init__(self,
        root_dir=data_dir,
        split='train',
        seq_len=8,
        sliding_step=1,
        data_mode='heatmap',
        bg_mode='',
        frame_alpha=-1,
        rally_i=None,
        rally_dir=None,
        frame_list=None,
        pred_dict=None,
        debug=False
    ):
        """ Shuttlecock_Trajectory_Dataset: https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw

            args:
                root_dir - A str of root directory path of dataset
                split - A str specifying the split mode, 'train', 'test', 'val'
                seq_len - A int specifying the length of input sequence
                sliding_step - A int specifying the step size of the sliding window when sampling inputs from the frame sequence
                data_mode - A str specifying the data mode, 'heatmap' or 'coordinate'
                bg_mode - A str specifying the background mode for TrackNet training, '', 'subtract', 'subtract_concat', 'concat'
                frame_alpha - A float specifying the alpha value of Beta distribution for frame mixup, -1 means no frame mixup
                rally_i - A int specifying the index of rally
                rally_dir - A str of frame directory path with format <data_dir>/<split>/match<match_id>/frame/<rally_id>
                frame_list - A list of images for TrackNet inference
                pred_dict - A dict of prediction results of TrackNet for InpaintNet inference
                debug - A bool specifying whether to use debug mode
            
        """
        assert split in ['train', 'test', 'val']
        assert data_mode in ['heatmap', 'coordinate']
        assert bg_mode in ['', 'subtract', 'subtract_concat', 'concat']

        # Image size
        self.HEIGHT = 288
        self.WIDTH = 512

        # Gaussian heatmap parameters
        self.mag = 1
        self.sigma = 2.5

        self.root_dir = root_dir
        self.split = split
        self.seq_len = seq_len
        self.sliding_step = sliding_step
        self.data_mode = data_mode
        self.bg_mode = bg_mode
        self.frame_alpha = frame_alpha

        # Data for inference
        self.frame_list = frame_list
        self.pred_dict = pred_dict

        # Initialize the input data
        if rally_dir is not None:
            if self.data_mode == 'heatmap':
                self.data_idxs, self.frame_files, self.coordinates, self.visibility = self._gen_input_from_rally_dir(rally_i, rally_dir)
            else:
                self.data_idxs, self.coordinates, self.pred_coordinates, self.visibility, self.pred_visibility, self.inpaint_mask = self._gen_input_from_rally_dir(rally_i, rally_dir)
        elif self.frame_list is not None:
            assert self.data_mode == 'heatmap'
            self.frame_list = np.array(frame_list)
            self.data_idxs = self._gen_input_from_frame_list()
            if self.bg_mode:
                self.median = np.median(np.array(frame_list), 0)
        elif self.pred_dict is not None:
            assert self.data_mode == 'coordinate'
            self.data_idxs, self.pred_coordinates, self.pred_visibility, self.inpaint_mask = self._gen_input_from_pred_dict()
        else:
            if self.data_mode == 'heatmap':
                self.input_file = os.path.join(self.root_dir, f'l{self.seq_len}_s{self.sliding_step}_{self.split}_{self.data_mode}.npz')
            else:
                self.input_file = os.path.join(self.root_dir, f'l{self.seq_len}_s{self.sliding_step}_{self.split}_{self.data_mode}.npz')
            
            if not os.path.exists(self.input_file):
                self._gen_input_file()
            data_dict = np.load(self.input_file)

            if debug:
                num_data = 256
                if self.data_mode == 'heatmap':
                    self.data_idxs = data_dict['data_idx'][:num_data]
                    self.frame_files = data_dict['filename'][:num_data]
                    self.coordinates = data_dict['coordinates'][:num_data]
                    self.visibility = data_dict['visibility'][:num_data]
                else:
                    self.data_idxs = data_dict['data_idx'][:num_data]
                    self.coordinates = data_dict['coordinates'][:num_data]
                    self.pred_coordinates = data_dict['pred_coordinates'][:num_data]
                    self.visibility = data_dict['visibility'][:num_data]
                    self.pred_visibility = data_dict['pred_visibility'][:num_data]
                    self.inpaint_mask = data_dict['inpaint_mask'][:num_data]
            else:
                if self.data_mode == 'heatmap':
                    self.data_idxs = data_dict['data_idx']                  # (N, F, 2)
                    self.frame_files = data_dict['filename']                # (N, F)
                    self.coordinates = data_dict['coordinates']             # (N, F, 2)
                    self.visibility = data_dict['visibility']               # (N, F)
                else:
                    self.data_idxs = data_dict['data_idx']                  # (N, F, 2)
                    self.coordinates = data_dict['coordinates']             # (N, F, 2)
                    self.pred_coordinates = data_dict['pred_coordinates']   # (N, F, 2)
                    self.visibility = data_dict['visibility']               # (N, F)
                    self.pred_visibility = data_dict['pred_visibility']     # (N, F)
                    self.inpaint_mask = data_dict['inpaint_mask']           # (N, F)

    def _gen_input_file(self):
        print('Generate input file...')
        rally_dirs = get_rally_dirs(self.root_dir, self.split)
        if self.data_mode == 'heatmap':
            data_idx = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
            frame_files = np.array([]).reshape(0, self.seq_len)
            coordinates = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            visibility = np.array([], dtype=np.float32).reshape(0, self.seq_len)

            # Generate input sequences from each rally
            for rally_i, rally_dir in enumerate(tqdm(rally_dirs)):
                rally_dir = os.path.join(self.root_dir, rally_dir)
                tmp_idx, tmp_frames, tmp_coor, tmp_vis = self._gen_input_from_rally_dir(rally_i, rally_dir)
                
                data_idx = np.concatenate((data_idx, tmp_idx), axis=0)
                frame_files = np.concatenate((frame_files, tmp_frames), axis=0)
                coordinates = np.concatenate((coordinates, tmp_coor), axis=0)
                visibility = np.concatenate((visibility, tmp_vis), axis=0)
            
            np.savez(self.input_file,
                    data_idx=data_idx,
                    filename=frame_files,
                    coordinates=coordinates,
                    visibility=visibility)
        else:
            data_idx = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
            coordinates = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            pred_coordinates = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            visibility = np.array([], dtype=np.float32).reshape(0, self.seq_len)
            pred_visibility = np.array([], dtype=np.float32).reshape(0, self.seq_len)
            inpaint_mask = np.array([], dtype=np.float32).reshape(0, self.seq_len)

            # Generate input sequences from each rally
            for rally_i, rally_dir in enumerate(tqdm(rally_dirs)):
                rally_dir = os.path.join(self.root_dir, rally_dir)
                tmp_idx, tmp_coor, tmp_coor_pred, tmp_vis, tmp_vis_pred, tmp_inpaint = self._gen_input_from_rally_dir(rally_i, rally_dir)
                data_idx = np.concatenate((data_idx, tmp_idx), axis=0)
                coordinates = np.concatenate((coordinates, tmp_coor), axis=0)
                pred_coordinates = np.concatenate((pred_coordinates, tmp_coor_pred), axis=0)
                visibility = np.concatenate((visibility, tmp_vis), axis=0)
                pred_visibility = np.concatenate((pred_visibility, tmp_vis_pred), axis=0)
                inpaint_mask = np.concatenate((inpaint_mask, tmp_inpaint), axis=0)
            
            np.savez(self.input_file,
                    data_idx=data_idx,
                    coordinates=coordinates,
                    pred_coordinates=pred_coordinates,
                    visibility=visibility,
                    pred_visibility=pred_visibility,
                    inpaint_mask=inpaint_mask)

    def _gen_input_from_rally_dir(self, rally_i, rally_dir):
        assert os.path.exists(rally_dir) and 'frame' in rally_dir
        match_dir, rally_id = parse.parse('{}/frame/{}', rally_dir)
        if self.data_mode == 'coordinate':
            csv_file = os.path.join(match_dir, 'predicted_csv', f'{rally_id}_ball.csv')
        else:
            if 'test' in rally_dir:
                csv_file = os.path.join(match_dir, 'corrected_csv', f'{rally_id}_ball.csv')
            else:
                csv_file = os.path.join(match_dir, 'csv', f'{rally_id}_ball.csv')
        
        try:
            label_df = pd.read_csv(csv_file, encoding='utf8').sort_values(by='Frame').fillna(0)
        except:
            raise Exception(f'{csv_file} does not exist.')

        if self.data_mode == 'heatmap':
            data_idx = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
            frame_files = np.array([]).reshape(0, self.seq_len)
            coordinates = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            visibility = np.array([], dtype=np.float32).reshape(0, self.seq_len)
            
            frame_file = np.array([f'{rally_dir}/{f_id}.png' for f_id in label_df['Frame']])
            x, y, vis = np.array(label_df['X']), np.array(label_df['Y']), np.array(label_df['Visibility'])
        
            # Sliding on the frame sequence
            for i in range(0, len(frame_file)-self.seq_len+1, self.sliding_step):
                tmp_idx, tmp_frames, tmp_coor, tmp_vis = [], [], [], []
                # Construct a single input sequence
                for f in range(self.seq_len):
                    if os.path.exists(frame_file[i+f]):
                        if rally_i:
                            tmp_idx.append((rally_i, i+f))
                        else:
                            tmp_idx.append((0, i+f))
                        tmp_frames.append(frame_file[i+f])
                        tmp_coor.append((x[i+f], y[i+f]))
                        tmp_vis.append(vis[i+f])
                    else:
                        break

                # Append the input sequence
                if len(tmp_frames) == self.seq_len:
                    assert len(tmp_frames) == len(tmp_coor) == len(tmp_vis)
                    data_idx = np.concatenate((data_idx, [tmp_idx]), axis=0)
                    frame_files = np.concatenate((frame_files, [tmp_frames]), axis=0)
                    coordinates = np.concatenate((coordinates, [tmp_coor]), axis=0)
                    visibility = np.concatenate((visibility, [tmp_vis]), axis=0)
            
            return data_idx, frame_files, coordinates, visibility
        else:
            data_idx = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
            coordinates = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            pred_coordinates = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            visibility = np.array([], dtype=np.float32).reshape(0, self.seq_len)
            pred_visibility = np.array([], dtype=np.float32).reshape(0, self.seq_len)
            inpaint_mask = np.array([], dtype=np.float32).reshape(0, self.seq_len)

            frame_file = np.array([f'{rally_dir}/{f_id}.png' for f_id in label_df['Frame']])
            x, y, vis = np.array(label_df['X']), np.array(label_df['Y']), np.array(label_df['Visibility'])
            x_pred, y_pred, vis_pred = np.array(label_df['X_pred']), np.array(label_df['Y_pred']), np.array(label_df['Visibility_pred'])
            inpaint = np.array(label_df['Inpainting'])

            # Sliding on the frame sequence
            for i in range(0, len(frame_file)-self.seq_len+1, self.sliding_step):
                tmp_idx, tmp_coor, tmp_coor_pred, tmp_vis, tmp_vis_pred, tmp_inpaint  = [], [], [], [], [], []
                # Construct a single input sequence
                for f in range(self.seq_len):
                    if os.path.exists(frame_file[i+f]):
                        if rally_i:
                            tmp_idx.append((rally_i, i+f))
                        else:
                            tmp_idx.append((0, i+f))
                        tmp_coor.append((x[i+f], y[i+f]))
                        tmp_coor_pred.append((x_pred[i+f], y_pred[i+f]))
                        tmp_vis.append(vis[i+f])
                        tmp_vis_pred.append(vis_pred[i+f])
                        tmp_inpaint.append(inpaint[i+f])
                    else:
                        break

                # Append the input sequence
                if len(tmp_idx) == self.seq_len:
                    assert len(tmp_idx) == len(tmp_coor) == len(tmp_coor_pred) == len(tmp_vis) == len(tmp_vis_pred) == len(tmp_inpaint)
                    data_idx = np.concatenate((data_idx, [tmp_idx]), axis=0)
                    coordinates = np.concatenate((coordinates, [tmp_coor]), axis=0)
                    pred_coordinates = np.concatenate((pred_coordinates, [tmp_coor_pred]), axis=0)
                    visibility = np.concatenate((visibility, [tmp_vis]), axis=0)
                    pred_visibility = np.concatenate((pred_visibility, [tmp_vis_pred]), axis=0)
                    inpaint_mask = np.concatenate((inpaint_mask, [tmp_inpaint]), axis=0)
            
            return data_idx, coordinates, pred_coordinates, visibility, pred_visibility, inpaint_mask

    def _gen_input_from_frame_list(self):
        data_idx = np.array([], dtype=np.int32).reshape(0, self.seq_len)

        for i in range(0, len(self.frame_list)-self.seq_len+1, self.sliding_step):
            tmp_idx = []
            # Construct a single input sequence
            for f in range(self.seq_len):
                tmp_idx.append(i+f)

            # Append the input sequence
            data_idx = np.concatenate((data_idx, [tmp_idx]), axis=0)
        
        return data_idx

    def _gen_input_from_pred_dict(self):
        data_idx = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
        pred_coordinates = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
        pred_visibility = np.array([], dtype=np.float32).reshape(0, self.seq_len)
        inpaint_mask = np.array([], dtype=np.float32).reshape(0, self.seq_len)
        x_pred, y_pred, vis_pred = self.pred_dict['X_pred'], self.pred_dict['Y_pred'], self.pred_dict['Visibility_pred']
        inpaint = self.pred_dict['Inpaint']
        assert len(x_pred) == len(y_pred) == len(vis_pred) == len(inpaint)

        # Sliding on the frame sequence
        for i in range(0, len(inpaint)-self.seq_len+1, self.sliding_step):
            tmp_idx, tmp_coor_pred, tmp_vis_pred, tmp_inpaint = [], [], [], []
            # Construct a single input sequence
            for f in range(self.seq_len):
                tmp_idx.append((0, i+f))
                tmp_coor_pred.append((x_pred[i+f], y_pred[i+f]))
                tmp_vis_pred.append(vis_pred[i+f])
                tmp_inpaint.append(inpaint[i+f])
                
            if len(tmp_idx) == self.seq_len:
                assert len(tmp_coor_pred) == len(tmp_inpaint)
                data_idx = np.concatenate((data_idx, [tmp_idx]), axis=0)
                pred_coordinates = np.concatenate((pred_coordinates, [tmp_coor_pred]), axis=0)
                pred_visibility = np.concatenate((pred_visibility, [tmp_vis_pred]), axis=0)
                inpaint_mask = np.concatenate((inpaint_mask, [tmp_inpaint]), axis=0)
        
        return data_idx, pred_coordinates, pred_visibility, inpaint_mask
    
    def _get_heatmap(self, cx, cy):
        if cx == cy == 0:
            return np.zeros((1, self.HEIGHT, self.WIDTH))
        x, y = np.meshgrid(np.linspace(1, self.WIDTH, self.WIDTH), np.linspace(1, self.HEIGHT, self.HEIGHT))
        heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
        heatmap[heatmap <= self.sigma**2] = 1.
        heatmap[heatmap > self.sigma**2] = 0.
        heatmap = heatmap * self.mag
        return heatmap.reshape(1, self.HEIGHT, self.WIDTH)

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, idx):
        if self.frame_list is not None:
            data_idx = self.data_idxs[idx]
            imgs = self.frame_list[data_idx]
            frames = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)

            # Get the resize scaler
            h, w, _ = imgs[0].shape
            h_ratio, w_ratio = h / self.HEIGHT, w / self.WIDTH

            if self.bg_mode:
                median_img = self.median[:, :, [2,1,0]]
            
            # Read image and generate heatmap
            for i in range(self.seq_len):
                img = Image.fromarray(imgs[i])
                if self.bg_mode == 'subtract':
                    img = Image.fromarray(np.sum(np.absolute(img - median_img), 2).astype('uint8'))
                    img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                    img = img.reshape(1, self.HEIGHT, self.WIDTH)
                elif self.bg_mode == 'subtract_concat':
                    diff_img = Image.fromarray(np.sum(np.absolute(img - median_img), 2).astype('uint8'))
                    diff_img = np.array(diff_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    diff_img = diff_img.reshape(1, self.HEIGHT, self.WIDTH)
                    img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                    img = np.moveaxis(img, -1, 0)
                    img = np.concatenate((img, diff_img), axis=0)
                else:
                    img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                    img = np.moveaxis(img, -1, 0)
                
                frames = np.concatenate((frames, img), axis=0)
            
            if self.bg_mode == 'concat':
                median_img = Image.fromarray(self.median.astype('uint8'))
                median_img = np.array(median_img.resize(size=(self.WIDTH, self.HEIGHT)))
                median_img = np.moveaxis(median_img, -1, 0)
                frames = np.concatenate((median_img, frames), axis=0)
            
            # Normalization
            frames /= 255.

            return frames

        elif self.pred_dict is not None:
            pred_coors = self.pred_coordinates[idx] # (F, 2)
            inpaint = self.inpaint_mask[idx] # (F,)
            
            # Normalization
            pred_coors[:, 0] = pred_coors[:, 0] / self.WIDTH
            pred_coors[:, 1] = pred_coors[:, 1] / self.HEIGHT

            return pred_coors, inpaint.reshape(-1, 1)

        elif self.data_mode == 'heatmap':
            if self.frame_alpha > 0:
                # Frame mixup
                lamb = np.random.beta(self.frame_alpha, self.frame_alpha)
                data_idx = self.data_idxs[idx]
                frame_file = self.frame_files[idx]
                coors = self.coordinates[idx]
                vis = self.visibility[idx]

                # Get the resize scaler
                w, h = Image.open(os.path.join(self.root_dir, frame_file[0])).size
                h_ratio, w_ratio = h / self.HEIGHT, w / self.WIDTH

                if self.bg_mode:
                    match_dir, rally_id, _ = parse.parse('{}/frame/{}/{}.png', frame_file[0])
                    median_file = os.path.join(self.root_dir, match_dir, 'median.npz') if os.path.exists(os.path.join(self.root_dir, match_dir, 'median.npz')) else os.path.join(self.root_dir, match_dir, 'frame', rally_id, 'median.npz')
                    assert os.path.exists(median_file)
                    median_img = np.load(median_file)['median'][:, :, [2,1,0]] # BGR -> RGB
                
                # Initialize the previous frame data
                prev_img = Image.open(os.path.join(self.root_dir, frame_file[0]))
                if self.bg_mode == 'subtract':
                    prev_img = Image.fromarray(np.sum(np.absolute(prev_img - median_img), 2).astype('uint8'))
                    prev_img = np.array(prev_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    prev_img = prev_img.reshape(1, self.HEIGHT, self.WIDTH)
                elif self.bg_mode == 'subtract_concat':
                    diff_img = Image.fromarray(np.sum(np.absolute(prev_img - median_img), 2).astype('uint8'))
                    diff_img = np.array(diff_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    diff_img = diff_img.reshape(1, self.HEIGHT, self.WIDTH)
                    prev_img = np.array(prev_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    prev_img = np.moveaxis(prev_img, -1, 0)
                    prev_img = np.concatenate((prev_img, diff_img), axis=0)
                else:
                    prev_img = np.array(prev_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    prev_img = np.moveaxis(prev_img, -1, 0)

                prev_coor = coors[0]
                prev_vis = vis[0]
                prev_heatmap = self._get_heatmap(int(coors[0][0]/ w_ratio), int(coors[0][1]/ h_ratio))
                
                # Keep first dimension as timestamp for resample
                if self.bg_mode == 'subtract':
                    frames = prev_img.reshape(1, 1, self.HEIGHT, self.WIDTH)
                elif self.bg_mode == 'subtract_concat':
                    frames = prev_img.reshape(1, 4, self.HEIGHT, self.WIDTH)
                else:
                    frames = prev_img.reshape(1, 3, self.HEIGHT, self.WIDTH)

                tmp_coors = prev_coor.reshape(1, -1)
                tmp_vis = prev_vis.reshape(1, -1)
                heatmaps = prev_heatmap
                
                # Read image and generate heatmap
                for i in range(1, self.seq_len):
                    cur_img = Image.open(os.path.join(self.root_dir, frame_file[i]))
                    if self.bg_mode == 'subtract':
                        cur_img = Image.fromarray(np.sum(np.absolute(cur_img - median_img), 2).astype('uint8'))
                        cur_img = np.array(cur_img.resize(size=(self.WIDTH, self.HEIGHT)))
                        cur_img = cur_img.reshape(1, self.HEIGHT, self.WIDTH)
                    elif self.bg_mode == 'subtract_concat':
                        diff_img = Image.fromarray(np.sum(np.absolute(cur_img - median_img), 2).astype('uint8'))
                        diff_img = np.array(diff_img.resize(size=(self.WIDTH, self.HEIGHT)))
                        diff_img = diff_img.reshape(1, self.HEIGHT, self.WIDTH)
                        cur_img = np.array(cur_img.resize(size=(self.WIDTH, self.HEIGHT)))
                        cur_img = np.moveaxis(cur_img, -1, 0)
                        cur_img = np.concatenate((cur_img, diff_img), axis=0)
                    else:
                        cur_img = np.array(cur_img.resize(size=(self.WIDTH, self.HEIGHT)))
                        cur_img = np.moveaxis(cur_img, -1, 0)

                    inter_img = prev_img * lamb + cur_img * (1 - lamb)

                    # Linear interpolation
                    if vis[i] == 0:
                        inter_coor = prev_coor
                        inter_vis = prev_vis
                        cur_heatmap = prev_heatmap
                        inter_heatmap = cur_heatmap
                    elif prev_vis == 0 or math.sqrt(pow(prev_coor[0]-coors[i][0], 2)+pow(prev_coor[1]-coors[i][1], 2)) < 10:
                        inter_coor = coors[i]
                        inter_vis = vis[i]
                        cur_heatmap = self._get_heatmap(int(inter_coor[0]/ w_ratio), int(inter_coor[1]/ h_ratio))
                        inter_heatmap = cur_heatmap
                    else:
                        inter_coor = coors[i]
                        inter_vis = vis[i]
                        cur_heatmap = self._get_heatmap(int(coors[i][0]/ w_ratio), int(coors[i][1]/ h_ratio))
                        inter_heatmap = prev_heatmap * lamb + cur_heatmap * (1 - lamb)
                    
                    tmp_coors = np.concatenate((tmp_coors, inter_coor.reshape(1, -1), coors[i].reshape(1, -1)), axis=0)
                    tmp_vis = np.concatenate((tmp_vis, np.array([inter_vis]).reshape(1, -1), np.array([vis[i]]).reshape(1, -1)), axis=0)
                    frames = np.concatenate((frames, inter_img[None,:,:,:], cur_img[None,:,:,:]), axis=0)
                    heatmaps = np.concatenate((heatmaps, inter_heatmap, cur_heatmap), axis=0)
                    
                    prev_img, prev_heatmap, prev_coor, prev_vis = cur_img, cur_heatmap, coors[i], vis[i]
                
                # Resample input sequence
                rand_id = np.random.choice(len(frames), self.seq_len, replace=False)
                rand_id = np.sort(rand_id)
                tmp_coors = tmp_coors[rand_id]
                tmp_vis = tmp_vis[rand_id]
                frames = frames[rand_id]
                heatmaps = heatmaps[rand_id]
                
                if self.bg_mode == 'concat':
                    median_img = Image.fromarray(median_img.astype('uint8'))
                    median_img = np.array(median_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    median_img = np.moveaxis(median_img, -1, 0)
                    frames = np.concatenate((median_img.reshape(1, 3, self.HEIGHT, self.WIDTH), frames), axis=0)
                
                # Reshape to model input format
                frames = frames.reshape(-1, self.HEIGHT, self.WIDTH)

                # Normalization
                frames /= 255.
                tmp_coors[:, 0] = tmp_coors[:, 0] / w
                tmp_coors[:, 1] = tmp_coors[:, 1] / h

                return data_idx, frames, heatmaps, tmp_coors, tmp_vis
            else:
                data_idx = self.data_idxs[idx]
                frame_file = self.frame_files[idx]
                coors = self.coordinates[idx]
                vis = self.visibility[idx]

                frames = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)
                heatmaps = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)

                # Get the resize scaler
                w, h = Image.open(os.path.join(self.root_dir, frame_file[0])).size
                h_ratio, w_ratio = h / self.HEIGHT, w / self.WIDTH
                
                # Read median image
                if self.bg_mode:
                    match_dir, rally_id, _ = parse.parse('{}/frame/{}/{}.png', frame_file[0])
                    median_file = os.path.join(self.root_dir, match_dir, 'median.npz') if os.path.exists(os.path.join(self.root_dir, match_dir, 'median.npz')) else os.path.join(self.root_dir, match_dir, 'frame', rally_id, 'median.npz')
                    assert os.path.exists(median_file)
                    median_img = np.load(median_file)['median'][:, :, [2,1,0]] # BGR -> RGB
                
                # Read image and generate heatmap
                for i in range(self.seq_len):
                    img = Image.open(os.path.join(self.root_dir, frame_file[i]))
                    if self.bg_mode == 'subtract':
                        img = Image.fromarray(np.sum(np.absolute(img - median_img), 2).astype('uint8'))
                        img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                        img = img.reshape(1, self.HEIGHT, self.WIDTH)
                    elif self.bg_mode == 'subtract_concat':
                        diff_img = Image.fromarray(np.sum(np.absolute(img - median_img), 2).astype('uint8'))
                        diff_img = np.array(diff_img.resize(size=(self.WIDTH, self.HEIGHT)))
                        diff_img = diff_img.reshape(1, self.HEIGHT, self.WIDTH)
                        img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                        img = np.moveaxis(img, -1, 0)
                        img = np.concatenate((img, diff_img), axis=0)
                    else:
                        img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                        img = np.moveaxis(img, -1, 0)
                    
                    frames = np.concatenate((frames, img), axis=0)
                    heatmap = self._get_heatmap(int(coors[i][0]/w_ratio), int(coors[i][1]/h_ratio))
                    heatmaps = np.concatenate((heatmaps, heatmap), axis=0)
                
                if self.bg_mode == 'concat':
                    median_img = Image.fromarray(median_img.astype('uint8'))
                    median_img = np.array(median_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    median_img = np.moveaxis(median_img, -1, 0)
                    frames = np.concatenate((median_img, frames), axis=0)

                # Normalization
                frames /= 255.
                coors[:, 0] = coors[:, 0] / w
                coors[:, 1] = coors[:, 1] / h

                return data_idx, frames, heatmaps, coors, vis
        elif self.data_mode == 'coordinate':
            data_idx = self.data_idxs[idx]
            coors = self.coordinates[idx] # (F, 2)
            pred_coors = self.pred_coordinates[idx] # (F, 2)
            vis = self.visibility[idx] # (F,)
            vis_pred = self.pred_visibility[idx] # (F,)
            inpaint = self.inpaint_mask[idx] # (F,)
            
            # Normalization
            coors[:, 0] = coors[:, 0] / self.WIDTH
            coors[:, 1] = coors[:, 1] / self.HEIGHT
            pred_coors[:, 0] = pred_coors[:, 0] / self.WIDTH
            pred_coors[:, 1] = pred_coors[:, 1] / self.HEIGHT

            return data_idx, pred_coors, coors, vis_pred.reshape(-1, 1), vis.reshape(-1, 1), inpaint.reshape(-1, 1)
        else:
            raise NotImplementedError
