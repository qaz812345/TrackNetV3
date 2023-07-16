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

data_dir = 'data'

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
        padding=False,
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
                frame_list - A list of images for TrackNet inference
                pred_dict - A dict of prediction results of TrackNet for InpaintNet inference
                debug - A bool specifying whether to use debug mode
            
        """
        super(Shuttlecock_Trajectory_Dataset, self).__init__()
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
        self.split = split if rally_dir is None else self._get_split(rally_dir)
        self.seq_len = seq_len
        self.sliding_step = sliding_step
        self.data_mode = data_mode
        self.bg_mode = bg_mode
        self.frame_alpha = frame_alpha
        self.rally_dict = self._get_rally_dict()

        # Data for inference
        self.rally_i = rally_i if rally_dir is None else self._get_rally_i(rally_dir)
        self.frame_list = frame_list
        self.pred_dict = pred_dict
        self.padding = padding and self.sliding_step == self.seq_len

        # Initialize the input data
        if self.frame_list is not None:
            # For TrackNet inference
            assert self.data_mode == 'heatmap'
            self.frame_list = frame_list
            self.data_dict = self._gen_input_from_frame_list()
            if self.bg_mode:
                self.median = np.median(frame_list, 0)
        elif self.pred_dict is not None:
            # For InpaintNet inference
            assert self.data_mode == 'coordinate'
            self.data_dict = self._gen_input_from_pred_dict()
        else:
            # For training and evaluation
            if self.rally_i is not None:
                # Rally based
                assert self.rally_i < len(self.rally_dict['i2p'])
                self.data_dict = self._gen_input_from_rally_dir(self.rally_i)
            else:
                # Split based
                # Generate and load input file 
                self.input_file = os.path.join(self.root_dir, f'l{self.seq_len}_s{self.sliding_step}_{self.split}_{self.data_mode}.npz')
                if not os.path.exists(self.input_file):
                    self._gen_input_file()
                self.data_dict = np.load(self.input_file)

            if debug:
                num_data = 256
                for key in data_dict.keys():
                    data_dict[key] = data_dict[key][:num_data]

    def _get_rally_dict(self):
        rally_dirs = get_rally_dirs(self.root_dir, self.split)
        rally_dict = {'i2p':{i: os.path.join(self.root_dir, rally_dir) for i, rally_dir in enumerate(rally_dirs)},
                      'p2i':{os.path.join(self.root_dir, rally_dir): i for i, rally_dir in enumerate(rally_dirs)}}
        return rally_dict

    def _get_rally_i(self, rally_dir):
        return self.rally_dict['p2i'][rally_dir]

    def _get_split(self, rally_dir):
        format_str = self.root_dir + '/{}/match{}'
        split, _ = parse.parse(format_str, rally_dir)
        return split
    
    def _get_image_scaler(self, rally_i=None):
        # (w_scaler, h_scaler)
        if rally_i is None:
            return self.data_dict['img_scaler']
        else:
            return self.data_dict['img_scaler'][rally_i] 

    def _get_image_shape(self, rally_i=None):
        # (w, h)
        if rally_i is None:
            return self.data_dict['img_shape']
        else:
            return self.data_dict['img_shape'][rally_i] 
    
    def _gen_input_file(self):
        print('Generate input file...')
        img_scaler = [] # (num_rally, 2)
        img_shape = [] # (num_rally, 2)
        if self.data_mode == 'heatmap':
            id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
            frame_file = np.array([]).reshape(0, self.seq_len)
            coor = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)

            # Generate input sequences from each rally
            for rally_i, rally_dir in tqdm(self.rally_dict['i2p'].items()):
                data_dict = self._gen_input_from_rally_dir(rally_i)
                id = np.concatenate((id, data_dict['id']), axis=0)
                frame_file = np.concatenate((frame_file, data_dict['frame_file']), axis=0)
                coor = np.concatenate((coor, data_dict['coor']), axis=0)
                vis = np.concatenate((vis, data_dict['vis']), axis=0)
                img_scaler.append(data_dict['img_scaler'])
                img_shape.append(data_dict['img_shape'])
            
            np.savez(self.input_file, id=id, frame_file=frame_file, coor=coor, vis=vis,
                     img_scaler=np.array(img_scaler), img_shape=np.array(img_shape))
        else:
            id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
            coor = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            coor_pred = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)
            pred_vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)
            inpaint_mask = np.array([], dtype=np.float32).reshape(0, self.seq_len)

            # Generate input sequences from each rally
            for rally_i, rally_dir in tqdm(self.rally_dict['i2p'].items()):
                data_dict = self._gen_input_from_rally_dir(rally_i)
                id = np.concatenate((id, data_dict['id']), axis=0)
                coor = np.concatenate((coor, data_dict['coor']), axis=0)
                coor_pred = np.concatenate((coor_pred, data_dict['coor_pred']), axis=0)
                vis = np.concatenate((vis, data_dict['vis']), axis=0)
                pred_vis = np.concatenate((pred_vis, data_dict['pred_vis']), axis=0)
                inpaint_mask = np.concatenate((inpaint_mask, data_dict['inpaint_mask']), axis=0)
                img_scaler.append(data_dict['img_scaler'])
                img_shape.append(data_dict['img_shape'])
            
            np.savez(self.input_file, id=id, coor=coor, coor_pred=coor_pred, vis=vis, pred_vis=pred_vis,
                    inpaint_mask=inpaint_mask, img_scaler=np.array(img_scaler), img_shape=np.array(img_shape))

    def _gen_input_from_rally_dir(self, rally_i):
        rally_dir = self.rally_dict['i2p'][rally_i]
        match_dir, rally_id = parse.parse('{}/frame/{}', rally_dir)

        # Calculate the image scaler
        w, h = Image.open(os.path.join(rally_dir, '0.png')).size
        h_scaler, w_scaler = h / self.HEIGHT, w / self.WIDTH

        if 'test' in rally_dir:
            csv_file = os.path.join(match_dir, 'corrected_csv', f'{rally_id}_ball.csv')
        else:
            csv_file = os.path.join(match_dir, 'csv', f'{rally_id}_ball.csv')
        try:
            label_df = pd.read_csv(csv_file, encoding='utf8').sort_values(by='Frame').fillna(0)
        except:
            raise Exception(f'{csv_file} does not exist.')
        
        if self.data_mode == 'heatmap':
            id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
            frame_file = np.array([]).reshape(0, self.seq_len)
            coor = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)
            
            f_file = np.array([os.path.join(rally_dir, f'{f_id}.png') for f_id in label_df['Frame']])
            x, y, v = np.array(label_df['X']), np.array(label_df['Y']), np.array(label_df['Visibility'])

            # Sliding on the frame sequence
            last_idx = -1
            for i in range(0, len(f_file), self.sliding_step):
                tmp_idx, tmp_frames, tmp_coor, tmp_vis = [], [], [], []
                # Construct a single input sequence
                for f in range(self.seq_len):
                    if i+f < len(f_file):
                        tmp_idx.append((rally_i, i+f))
                        tmp_frames.append(f_file[i+f])
                        tmp_coor.append((x[i+f], y[i+f]))
                        tmp_vis.append(v[i+f])
                        last_idx = i+f
                    else:
                        if self.padding:
                            tmp_idx.append((rally_i, last_idx))
                            tmp_frames.append(f_file[last_idx])
                            tmp_coor.append((x[last_idx], y[last_idx]))
                            tmp_vis.append(v[last_idx])
                
                # Append the input sequence
                if len(tmp_frames) == self.seq_len:
                    assert len(tmp_frames) == len(tmp_coor) == len(tmp_vis)
                    id = np.concatenate((id, [tmp_idx]), axis=0)
                    frame_file = np.concatenate((frame_file, [tmp_frames]), axis=0)
                    coor = np.concatenate((coor, [tmp_coor]), axis=0)
                    vis = np.concatenate((vis, [tmp_vis]), axis=0)
            
            return dict(id=id, frame_file=frame_file, coor=coor, vis=vis,
                        img_scaler=(w_scaler, h_scaler), img_shape=(w, h))
        else:
            id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
            coor = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            coor_pred = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)
            pred_vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)
            inpaint_mask = np.array([], dtype=np.float32).reshape(0, self.seq_len)

            pred_csv_file = os.path.join(match_dir, 'predicted_csv', f'{rally_id}_ball.csv')
            try:
                pred_df = pd.read_csv(pred_csv_file, encoding='utf8').sort_values(by='Frame').fillna(0)
            except:
                raise Exception(f'{csv_file} does not exist.')
            assert len(label_df) == len(pred_df)
            f_file = np.array([os.path.join(rally_dir, f'{f_id}.png') for f_id in label_df['Frame']])
            x, y, v = np.array(label_df['X']), np.array(label_df['Y']), np.array(label_df['Visibility'])
            x_pred, y_pred, v_pred = np.array(pred_df['X']), np.array(pred_df['Y']), np.array(pred_df['Visibility'])
            inpaint = np.array(pred_df['Inpainting'])

            # Sliding on the frame sequence
            last_idx = -1
            for i in range(0, len(f_file), self.sliding_step):
                tmp_idx, tmp_coor, tmp_coor_pred, tmp_vis, tmp_vis_pred, tmp_inpaint  = [], [], [], [], [], []
                # Construct a single input sequence
                for f in range(self.seq_len):
                    if i+f < len(f_file):
                        tmp_idx.append((rally_i, i+f))
                        tmp_coor.append((x[i+f], y[i+f]))
                        tmp_coor_pred.append((x_pred[i+f], y_pred[i+f]))
                        tmp_vis.append(v[i+f])
                        tmp_vis_pred.append(v_pred[i+f])
                        tmp_inpaint.append(inpaint[i+f])
                    else:
                        if self.padding:
                            tmp_idx.append((rally_i, last_idx))
                            tmp_coor.append((x[last_idx], y[last_idx]))
                            tmp_coor_pred.append((x_pred[last_idx], y_pred[last_idx]))
                            tmp_vis.append(v[last_idx])
                            tmp_vis_pred.append(v_pred[last_idx])
                            tmp_inpaint.append(inpaint[last_idx])

                # Append the input sequence
                if len(tmp_idx) == self.seq_len:
                    assert len(tmp_idx) == len(tmp_coor) == len(tmp_coor_pred) == len(tmp_vis) == len(tmp_vis_pred) == len(tmp_inpaint)
                    id = np.concatenate((id, [tmp_idx]), axis=0)
                    coor = np.concatenate((coor, [tmp_coor]), axis=0)
                    coor_pred = np.concatenate((coor_pred, [tmp_coor_pred]), axis=0)
                    vis = np.concatenate((vis, [tmp_vis]), axis=0)
                    pred_vis = np.concatenate((pred_vis, [tmp_vis_pred]), axis=0)
                    inpaint_mask = np.concatenate((inpaint_mask, [tmp_inpaint]), axis=0)
            
            return dict(id=id, coor=coor, coor_pred=coor_pred, vis=vis, pred_vis=pred_vis, inpaint_mask=inpaint_mask,
                        img_scaler=(w_scaler, h_scaler), img_shape=(w, h))

    def _gen_input_from_frame_list(self):
        # Calculate the image scaler
        h, w, _ = self.frame_list[0].shape
        h_scaler, w_scaler = h / self.HEIGHT, w / self.WIDTH

        id = np.array([], dtype=np.int32).reshape(0, self.seq_len)
        last_idx = -1
        for i in range(0, len(self.frame_list), self.sliding_step):
            tmp_idx = []
            # Construct a single input sequence
            for f in range(self.seq_len):
                if i+f < len(self.frame_list):
                    tmp_idx.append(i+f)
                    last_idx = i+f
                else:
                    if self.padding:
                        tmp_idx.append(last_idx)

            # Append the input sequence
            id = np.concatenate((id, [tmp_idx]), axis=0)
        
        return dict(id=id, img_scaler=[(w_scaler, h_scaler)])

    def _gen_input_from_pred_dict(self):
        id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
        coor_pred = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
        pred_vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)
        inpaint_mask = np.array([], dtype=np.float32).reshape(0, self.seq_len)
        x_pred, y_pred, vis_pred = self.pred_dict['X_pred'], self.pred_dict['Y_pred'], self.pred_dict['Visibility_pred']
        inpaint = self.pred_dict['Inpaint']
        assert len(x_pred) == len(y_pred) == len(vis_pred) == len(inpaint)
        
        # Sliding on the frame sequence
        last_idx = -1
        for i in range(0, len(inpaint), self.sliding_step):
            tmp_idx, tmp_coor_pred, tmp_vis_pred, tmp_inpaint = [], [], [], []
            # Construct a single input sequence
            for f in range(self.seq_len):
                if i+f < len(inpaint):
                    tmp_idx.append((0, i+f))
                    tmp_coor_pred.append((x_pred[i+f], y_pred[i+f]))
                    tmp_vis_pred.append(vis_pred[i+f])
                    tmp_inpaint.append(inpaint[i+f])
                    last_idx = i+f
                else:
                    if self.padding:
                        tmp_idx.append((0, last_idx))
                        tmp_coor_pred.append((x_pred[last_idx], y_pred[last_idx]))
                        tmp_vis_pred.append(vis_pred[last_idx])
                        tmp_inpaint.append(inpaint[last_idx])
                
            if len(tmp_idx) == self.seq_len:
                assert len(tmp_coor_pred) == len(tmp_inpaint)
                id = np.concatenate((id, [tmp_idx]), axis=0)
                coor_pred = np.concatenate((coor_pred, [tmp_coor_pred]), axis=0)
                pred_vis = np.concatenate((pred_vis, [tmp_vis_pred]), axis=0)
                inpaint_mask = np.concatenate((inpaint_mask, [tmp_inpaint]), axis=0)
        
        return dict(id=id, coor_pred=coor_pred, pred_vis=pred_vis, inpaint_mask=inpaint_mask,
                    img_scaler=self.pred_dict['Img_scaler'], img_shape=self.pred_dict['Img_shape']) 
    
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
        return len(self.data_dict['id'])

    def __getitem__(self, idx):
        if self.frame_list is not None:
            data_idx = self.data_dict['id'][idx] # (L,)
            imgs = self.frame_list[data_idx] # (L, H, W, 3)
            median_img = self.median
            
            # Process frame sequence
            frames = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)
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
            coor_pred = self.data_dict['coor_pred'][idx] # (L, 2)
            inpaint = self.data_dict['inpaint_mask'][idx].reshape(-1, 1) # (L, 1)
            w, h = self.data_dict['img_shape']
            
            # Normalization
            coor_pred[:, 0] = coor_pred[:, 0] / w
            coor_pred[:, 1] = coor_pred[:, 1] / h

            return coor_pred, inpaint

        elif self.data_mode == 'heatmap':
            if self.frame_alpha > 0:
                # Frame mixup
                lamb = np.random.beta(self.frame_alpha, self.frame_alpha)
                data_idx = self.data_dict['id'][idx] # (L,)
                frame_file = self.data_dict['frame_file'][idx] # (L,)
                coor = self.data_dict['coor'][idx] # (L, 2)
                vis = self.data_dict['vis'][idx] # (L,)
                w, h = self._get_image_shape(data_idx[0][0]) if self.rally_i is None else self.data_dict['img_shape']
                w_scaler, h_scaler = self._get_image_scaler(data_idx[0][0]) if self.rally_i is None else self.data_dict['img_scaler']

                if self.bg_mode:
                    match_dir, rally_id, _ = parse.parse('{}/frame/{}/{}.png', frame_file[0])
                    median_file = os.path.join(match_dir, 'median.npz') if os.path.exists(os.path.join(match_dir, 'median.npz')) else os.path.join(match_dir, 'frame', rally_id, 'median.npz')
                    assert os.path.exists(median_file)
                    median_img = np.load(median_file)['median']
                
                # Initialize the previous frame data
                prev_img = Image.open(frame_file[0])
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

                prev_coor = coor[0]
                prev_vis = vis[0]
                prev_heatmap = self._get_heatmap(int(coor[0][0]/ w_scaler), int(coor[0][1]/ h_scaler))
                
                # Keep first dimension as timestamp for resample
                if self.bg_mode == 'subtract':
                    frames = prev_img.reshape(1, 1, self.HEIGHT, self.WIDTH)
                elif self.bg_mode == 'subtract_concat':
                    frames = prev_img.reshape(1, 4, self.HEIGHT, self.WIDTH)
                else:
                    frames = prev_img.reshape(1, 3, self.HEIGHT, self.WIDTH)

                tmp_coor = prev_coor.reshape(1, -1)
                tmp_vis = prev_vis.reshape(1, -1)
                heatmaps = prev_heatmap
                
                # Read image and generate heatmap
                for i in range(1, self.seq_len):
                    cur_img = Image.open(frame_file[i])
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
                    elif prev_vis == 0 or math.sqrt(pow(prev_coor[0]-coor[i][0], 2)+pow(prev_coor[1]-coor[i][1], 2)) < 10:
                        inter_coor = coor[i]
                        inter_vis = vis[i]
                        cur_heatmap = self._get_heatmap(int(inter_coor[0]/ w_scaler), int(inter_coor[1]/ h_scaler))
                        inter_heatmap = cur_heatmap
                    else:
                        inter_coor = coor[i]
                        inter_vis = vis[i]
                        cur_heatmap = self._get_heatmap(int(coor[i][0]/ w_scaler), int(coor[i][1]/ h_scaler))
                        inter_heatmap = prev_heatmap * lamb + cur_heatmap * (1 - lamb)
                    
                    tmp_coor = np.concatenate((tmp_coor, inter_coor.reshape(1, -1), coor[i].reshape(1, -1)), axis=0)
                    tmp_vis = np.concatenate((tmp_vis, np.array([inter_vis]).reshape(1, -1), np.array([vis[i]]).reshape(1, -1)), axis=0)
                    frames = np.concatenate((frames, inter_img[None,:,:,:], cur_img[None,:,:,:]), axis=0)
                    heatmaps = np.concatenate((heatmaps, inter_heatmap, cur_heatmap), axis=0)
                    
                    prev_img, prev_heatmap, prev_coor, prev_vis = cur_img, cur_heatmap, coor[i], vis[i]
                
                # Resample input sequence
                rand_id = np.random.choice(len(frames), self.seq_len, replace=False)
                rand_id = np.sort(rand_id)
                tmp_coor = tmp_coor[rand_id]
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
                tmp_coor[:, 0] = tmp_coor[:, 0] / w
                tmp_coor[:, 1] = tmp_coor[:, 1] / h

                return data_idx, frames, heatmaps, tmp_coor, tmp_vis
            else:
                data_idx = self.data_dict['id'][idx]
                frame_file = self.data_dict['frame_file'][idx]
                coor = self.data_dict['coor'][idx]
                vis = self.data_dict['vis'][idx]
                w, h = self._get_image_shape(data_idx[0][0]) if self.rally_i is None else self.data_dict['img_shape']
                w_scaler, h_scaler = self._get_image_scaler(data_idx[0][0]) if self.rally_i is None else self.data_dict['img_scaler']

                # Read median image
                if self.bg_mode:
                    match_dir, rally_id, _ = parse.parse('{}/frame/{}/{}.png', frame_file[0])
                    median_file = os.path.join(match_dir, 'median.npz') if os.path.exists(os.path.join(match_dir, 'median.npz')) else os.path.join(match_dir, 'frame', rally_id, 'median.npz')
                    assert os.path.exists(median_file)
                    median_img = np.load(median_file)['median']

                frames = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)
                heatmaps = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)
                
                # Read image and generate heatmap
                for i in range(self.seq_len):
                    img = Image.open(frame_file[i])
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
                    
                    heatmap = self._get_heatmap(int(coor[i][0]/w_scaler), int(coor[i][1]/h_scaler))
                    frames = np.concatenate((frames, img), axis=0)
                    heatmaps = np.concatenate((heatmaps, heatmap), axis=0)
                
                if self.bg_mode == 'concat':
                    median_img = Image.fromarray(median_img.astype('uint8'))
                    median_img = np.array(median_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    median_img = np.moveaxis(median_img, -1, 0)
                    frames = np.concatenate((median_img, frames), axis=0)

                # Normalization
                frames /= 255.
                coor[:, 0] = coor[:, 0] / w
                coor[:, 1] = coor[:, 1] / h

                return data_idx, frames, heatmaps, coor, vis
        
        elif self.data_mode == 'coordinate':
            data_idx = self.data_dict['id'][idx] # (L,)
            coor = self.data_dict['coor'][idx] # (L, 2)
            coor_pred = self.data_dict['coor_pred'][idx] # (L, 2)
            vis = self.data_dict['vis'][idx] # (L,)
            vis_pred = self.data_dict['pred_vis'][idx] # (L,)
            inpaint = self.data_dict['inpaint_mask'][idx] # (L,)
            w, h = self._get_image_shape(data_idx[0][0]) if self.rally_i is None else self.data_dict['img_shape']
            
            # Normalization
            coor[:, 0] = coor[:, 0] / w
            coor[:, 1] = coor[:, 1] / h
            coor_pred[:, 0] = coor_pred[:, 0] / w
            coor_pred[:, 1] = coor_pred[:, 1] / h

            return data_idx, coor_pred, coor, vis_pred.reshape(-1, 1), vis.reshape(-1, 1), inpaint.reshape(-1, 1)
        else:
            raise NotImplementedError
