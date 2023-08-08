import os
import json
import queue
import numpy as np
import multiprocessing as mp
import subprocess as sp
from utils import *

# 1. How many workers do you want for each GPU?
# ---------------------------------------------
# E.g. you have 2 RTX 2080 (8GB) and 2 RTX 2080 Ti (11GB),
# configure the number of processes to be run in parallel for each GPU.
CUDA_WORKERS = {
    #0: 3, # 3 workers for CUDA_VISIBLE_DEVICES=0
    1: 1, # 3 workers for CUDA_VISIBLE_DEVICES=1
    #2: 3, # 4 workers for CUDA_VISIBLE_DEVICES=2
    #3: 3, # 4 workers for CUDA_VISIBLE_DEVICES=3
    #4: 3, # 4 workers for CUDA_VISIBLE_DEVICES=4 
    #5: 3, # 4 workers for CUDA_VISIBLE_DEVICES=5
}

# 2. List of commands you want to run
# ---------------------------------------------
# pick training points by uniform sampling

# python test.py --tracknet_file TrackNet_best.pt --inpaintnet_file InpaintNet_best.pt --save_dir eval

dirs = ['test/wo_skip']

exp_dirs = []
for d in dirs:
    for dir_path, dir_names, file_name in os.walk(d):
        #print(dirPath)
        for f in file_name:
            if 'InpaintNet_best.pt' in f:
                exp_dirs.append(dir_path)

COMMANDS = [
    ['python3', 'test.py',
    '--tracknet_file', 'ckpts/TrackNetV3/TrackNet_best.pt',
    '--inpaintnet_file', f'{exp_dir}/InpaintNet_best.pt',
    '--save_dir', f'{exp_dir}/eval'
    ]
    for exp_dir in exp_dirs
]

'''       
COMMANDS = [
    ['python3', 'train.py',
    '--model_name', 'InpaintNet',
    '--seq_len', str(l),
    '--epoch', '300',
    '--batch_size', '32',
    '--lr_scheduler', 'StepLR',
    '--mask_ratio', str(mr),
    '--save_dir', f'test/wo_skip/l{l}_mask{mr}'
    ]
    for l in [8, 16, 32]
    for mr in [0.2, 0.3, 0.4]
]
'''



# or:
# COMMANDS = [
#     'python3 train.py --dataset=./dataset/MNIST --lr=0.003',
#     'python3 train.py --dataset=./dataset/MNIST --lr=0.005',
#     'python3 train.py --dataset=./dataset/MNIST --lr=0.007',
#     'python3 train.py --dataset=./dataset/MNIST --lr=0.003',
# ]


def worker(cuda_no, worker_no, cmd_queue):
    worker_name = 'CUDA-{}:{}'.format(cuda_no, worker_no)
    print(worker_name, 'started')
    
    env = os.environ.copy()
    # overwrite visible cuda devices
    env['CUDA_VISIBLE_DEVICES'] = str(cuda_no)#'1,3'#
    
    while True:
        cmd = cmd_queue.get()
        
        if cmd is None:
            cmd_queue.task_done()
            break
        
        print(worker_name, cmd)
        
        shell = {str: True, list: False}.get(type(cmd))
        assert shell is not None, 'cmd should be list or str'
        
        sp.Popen(cmd, shell=shell, env=env).wait()
        cmd_queue.task_done()
    
    print(worker_name, 'stopped')


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    cmd_queue = mp.JoinableQueue()
    
    for cmd in COMMANDS:
        cmd_queue.put(cmd)
        
    for _ in range(sum(CUDA_WORKERS.values())):
        # workers stop after getting None
        cmd_queue.put(None)
        
    procs = [
        mp.Process(target=worker, args=(cuda_no, worker_no, cmd_queue), daemon=True)
        for cuda_no, num_workers in CUDA_WORKERS.items()
        for worker_no in range(num_workers)
    ]
    
    for proc in procs:
        proc.start()

    cmd_queue.join()
        
    for proc in procs:
        proc.join()