import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import Shuttlecock_Trajectory_Dataset
from test import eval_tracknet, eval_inpaintnet
from utils.general import ResumeArgumentParser, get_model, to_img_format
from utils.metric import WBCELoss
from utils.visualize import plot_heatmap_pred_sample, plot_traj_pred_sample, write_to_tb


def mixup(x, y, alpha=0.5):
    """Returns mixed inputs, pairs of targets.
    
        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Target tensor
            alpha (float): Alpha of beta distribution

        Returns:
            x_mix (torch.Tensor): Mixed input tensor
            y_mix (torch.Tensor): Mixed target tensor
    """

    batch_size = x.size()[0]
    lamb = np.random.beta(alpha, alpha, size=batch_size)
    lamb = np.maximum(lamb, 1 - lamb)
    lamb = torch.from_numpy(lamb[:, None, None, None]).float().to(x.device)
    index = torch.randperm(batch_size)
    x_mix = x * lamb + x[index] * (1 - lamb)
    y_mix = y * lamb + y[index] * (1 - lamb)

    return x_mix, y_mix

def get_random_mask(mask_size, mask_ratio):
    """ Generate random mask by binomial distribution.
        1 means masked, 0 means not.
    
        Args:
            mask_size (Tuple[int, int]): Mask size (N, L)
            mask_ratio (float): Ratio of masked area

        Returns:
            mask (torch.Tensor): Random mask tensor with shape (N, L, 1)
    """

    mask = np.random.binomial(1, mask_ratio, size=mask_size)
    mask = torch.from_numpy(mask).float().cuda().unsqueeze(-1)

    return mask

def train_tracknet(model, optimizer, data_loader, param_dict):
    """ Train TrackNet model for one epoch.

        Args:
            model (torch.nn.Module): TrackNet model
            optimizer (torch.optim): Optimizer
            data_loader (torch.utils.data.DataLoader): Data loader
            param_dict (Dict): Parameters
                - param_dict['alpha'] (float): Alpha of sample mixup
                - param_dict['verbose'] (bool): Control whether to show progress bar
                - param_dict['bg_mode'] (str): For visualizing current prediction
                - param_dict['save_dir'] (str): For saving current prediction
        
        Returns:
            (float): Average loss
    """

    model.train()
    epoch_loss = []

    if param_dict['verbose']:
        data_prob = tqdm(train_loader)
    else:
        data_prob = data_loader
    
    for step, (_, x, y, c, _) in enumerate(data_prob):
        optimizer.zero_grad()
        x, y = x.float().cuda(), y.float().cuda()

        # Sample mixup
        if param_dict['alpha'] > 0:
            x, y = mixup(x, y, param_dict['alpha'])
        
        y_pred = model(x)
        loss = WBCELoss(y_pred, y)
        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if param_dict['verbose'] and (step + 1) % display_step == 0:
            data_prob.set_description(f'Training')
            data_prob.set_postfix(loss=loss.item())

        # Visualize current prediction
        if (step + 1) % display_step == 0:
            x, y, y_pred = x.detach().cpu().numpy(), y.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
            c = c.numpy()
            
            # Transform inputs to image format (N, L, H , W, C)
            if param_dict['bg_mode'] == 'subtract':
                x = to_img_format(x)
            elif param_dict['bg_mode'] == 'subtract_concat':
                x = to_img_format(x, num_ch=4)
            elif param_dict['bg_mode'] == 'concat':
                x = to_img_format(x, num_ch=3)
                x = x[:, 1:, :, :, :]
            else:
                x = to_img_format(x, num_ch=3)
            y = to_img_format(y)
            y_pred = to_img_format(y_pred)
            plot_heatmap_pred_sample(x[0], y[0], y_pred[0], c[0], bg_mode=param_dict['bg_mode'], save_dir=param_dict['save_dir'])
    
    return float(np.mean(epoch_loss))
   
def train_inpaintnet(model, optimizer, data_loader, param_dict):
    """ Train InpaintNet model for one epoch.

        Args:
            model (torch.nn.Module): InpaintNet model
            optimizer (torch.optim): Optimizer
            data_loader (torch.utils.data.DataLoader): Data loader
            param_dict (Dict): parameters
                - param_dict['verbose'] (bool): Control whether to show progress bar
                - param_dict['mask_ratio'] (float): Ratio of masked area
                - param_dict['save_dir'] (str): For saving current prediction

        Returns:
            (float): Average loss
    """

    model.train()
    epoch_loss = []

    if param_dict['verbose']:
        data_prob = tqdm(data_loader)
    else:
        data_prob = data_loader

    for step, (_, coor_pred, coor_gt, _, vis_gt, _) in enumerate(data_prob):
        optimizer.zero_grad()
        coor_pred, coor_gt, vis_gt = coor_pred.float().cuda(), coor_gt.float().cuda(), vis_gt.float().cuda()

        # Sample random mask as inpainting mask
        mask = get_random_mask(mask_size=coor_gt.shape[:2], mask_ratio=param_dict['mask_ratio']).cuda() # (N, L, 1)
        inpaint_mask = torch.logical_and(vis_gt, mask).int() # visible and masked area
        
        coor_pred = coor_pred * (1 - inpaint_mask) # masked area is set to 0
        refine_coor = model(coor_pred, inpaint_mask)

        # Calculate masked loss
        masked_refine_coor = refine_coor * inpaint_mask
        masked_gt_coor = coor_gt * inpaint_mask
        loss = nn.MSELoss()(masked_refine_coor, masked_gt_coor)
        epoch_loss.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        
        if param_dict['verbose'] and (step + 1) % display_step == 0:
            data_prob.set_description(f'Training')
            data_prob.set_postfix(loss=loss.item())

        # Visualize current prediction
        if (step + 1) % display_step == 0:
            coor_gt, refine_coor, inpaint_mask = coor_gt.detach().cpu().numpy(), refine_coor.detach().cpu().numpy(), inpaint_mask.detach().cpu().numpy()
            plot_traj_pred_sample(coor_gt[0], refine_coor[0], inpaint_mask[0], save_dir=param_dict['save_dir'])
    
    return float(np.mean(epoch_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='TrackNet', choices=['TrackNet', 'InpaintNet'], help='model type')
    parser.add_argument('--seq_len', type=int, default=8, help='sequence length of input')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size of training')
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD', 'Adadelta'], help='optimizer')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='', choices=['', 'StepLR'], help='learning rate scheduler')
    parser.add_argument('--bg_mode', type=str, default='', choices=['', 'subtract', 'subtract_concat', 'concat'], help='background mode')
    parser.add_argument('--alpha', type=float, default=-1, help='alpha of sample mixup, -1 means no mixup')
    parser.add_argument('--frame_alpha', type=float, default=-1, help='alpha of frame mixup, -1 means no mixup')
    parser.add_argument('--mask_ratio', type=float, default=0.3, help='ratio of random mask during training InpaintNet')
    parser.add_argument('--tolerance', type=float, default=4, help='difference tolerance of center distance between prediction and ground truth in input size')
    parser.add_argument('--resume_training', action='store_true', default=False, help='resume training from experiment directory')
    parser.add_argument('--seed', type=int, default=13, help='random seed')
    parser.add_argument('--save_dir', type=str, default='exp', help='directory to save the checkpoints and prediction result')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    param_dict = vars(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    print(f"TensorBoard: start with 'tensorboard --logdir {os.path.join(args.save_dir, 'logs')}', view at http://localhost:6006/")
    tb_writer = SummaryWriter(os.path.join(args.save_dir, 'logs'))

    display_step = 4 if args.debug else 100
    num_workers = args.batch_size if args.batch_size <= 16 else 16
    
    # Load checkpoint
    if args.resume_training:
        print(f'Load checkpoint from {args.model_name}_cur.pt...')
        assert os.path.exists(os.path.join(args.save_dir, f'{args.model_name}_cur.pt')), f'No checkpoint found in {args.save_dir}'
        ckpt = torch.load(os.path.join(args.save_dir, f'{args.model_name}_cur.pt'))
        param_dict = ckpt['param_dict']
        ckpt['param_dict']['resume_training'] = args.resume_training
        ckpt['param_dict']['epochs'] = args.epochs
        ckpt['param_dict']['verbose'] = args.verbose
        args = ResumeArgumentParser(ckpt['param_dict'])

    print(f'Parameters: {param_dict}')
    print(f'Load dataset...')
    data_mode = 'heatmap' if args.model_name == 'TrackNet' else 'coordinate'
    train_dataset = Shuttlecock_Trajectory_Dataset(split='train', seq_len=args.seq_len, sliding_step=1, data_mode=data_mode, bg_mode=args.bg_mode, frame_alpha=args.frame_alpha, debug=args.debug)
    val_dataset = Shuttlecock_Trajectory_Dataset(split='val', seq_len=args.seq_len, sliding_step=args.seq_len, data_mode=data_mode, bg_mode=args.bg_mode, debug=args.debug)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=True)

    print(f'Create {args.model_name}...')
    model = get_model(args.model_name, args.seq_len, args.bg_mode).cuda() if args.model_name == 'TrackNet' else get_model(args.model_name).cuda()
    train_fn = train_tracknet if args.model_name == 'TrackNet' else train_inpaintnet
    eval_fn = eval_tracknet if args.model_name == 'TrackNet' else eval_inpaintnet

    # Create optimizer
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optim == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError('Invalid optimizer.')

    # Create lr scheduler
    if args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.epochs/3), gamma=0.1)
    else:
        scheduler = None

    # Init statistics
    if args.resume_training:
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if args.lr_scheduler:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        max_val_acc = ckpt['max_val_acc']
        print(f'Resume training from epoch {start_epoch}...')
    else:
        max_val_acc = 0.
        start_epoch = 0
        

    print(f'Start training...')
    train_start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        print(f'Epoch [{epoch+1} / {args.epochs}]')
        start_time = time.time()
        train_loss = train_fn(model, optimizer, train_loader, param_dict)
        val_loss, val_res = eval_fn(model, val_loader, param_dict)
        write_to_tb(args.model_name, tb_writer, (train_loss, val_loss), val_res, epoch)

        if args.lr_scheduler:
            scheduler.step()
        
        # Pick best model
        cur_val_acc = val_res['accuracy'] if args.model_name == 'TrackNet' else val_res['inpaint']['accuracy']
        if cur_val_acc >= max_val_acc:
            max_val_acc = cur_val_acc
            torch.save(dict(epoch=epoch,
                            max_val_acc=max_val_acc,
                            model=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict() if scheduler is not None else None,
                            param_dict=param_dict),
                        os.path.join(args.save_dir, f'{args.model_name}_best.pt'))
        
        # Save current model
        torch.save(dict(epoch=epoch,
                        max_val_acc=max_val_acc,
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict() if scheduler is not None else None,
                        param_dict=param_dict),
                    os.path.join(args.save_dir, f'{args.model_name}_cur.pt'))
        
        print(f'Epoch runtime: {(time.time() - start_time) / 3600.:.2f} hrs')
    
    tb_writer.close()
    print(f'Training time: {(time.time() - train_start_time) / 3600.:.2f} hrs')
    print('Done......')