import os
import parse
import shutil

from dataset import data_dir
from utils.general import list_dirs, generate_frames, get_num_frames, get_match_median

# Form validation set
if not os.path.exists(os.path.join(data_dir, 'val')):
    match_dirs = list_dirs(os.path.join(data_dir, 'train'))
    match_dirs = sorted(match_dirs, key=lambda s: int(s.split('match')[-1]))
    for match_dir in match_dirs:
        video_files = list_dirs(os.path.join(match_dir, 'video'))
        _, match_dir, rally_id = parse.parse('{}/train/{}/video/{}.mp4', video_files[-1]) # pick last rally
        os.makedirs(os.path.join(data_dir, 'val', match_dir, 'csv'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'val', match_dir, 'video'), exist_ok=True)
        shutil.move(os.path.join(data_dir, 'train', match_dir, 'csv', f'{rally_id}_ball.csv'),
                    os.path.join(data_dir, 'val', match_dir, 'csv', f'{rally_id}_ball.csv'))
        shutil.move(os.path.join(data_dir, 'train', match_dir, 'video', f'{rally_id}.mp4'),
                    os.path.join(data_dir, 'val', match_dir, 'video', f'{rally_id}.mp4'))

# Replace csv to corrected csv in test set
if os.path.exists('corrected_test_label'):
    match_dirs = list_dirs(os.path.join(data_dir, 'test'))
    match_dirs = sorted(match_dirs, key=lambda s: int(s.split('match')[-1]))
    for match_dir in match_dirs:
        _, match_dir = parse.parse('{}/test/{}', match_dir)
        shutil.copytree(os.path.join('corrected_test_label', match_dir, 'corrected_csv'),
                        os.path.join(data_dir, 'test', match_dir, 'corrected_csv'))


# Generate frames
for split in ['train', 'val', 'test']:
    split_frame_count = 0
    match_dirs = list_dirs(os.path.join(data_dir, split))
    for match_dir in match_dirs:
        match_frame_count = 0
        match_name = match_dir.split('/')[-1]
        video_files = list_dirs(os.path.join(match_dir, 'video'))
        for video_file in video_files:
            generate_frames(video_file)
            video_frame_count = get_num_frames(video_file)
            video_name = video_file.split('/')[-1]
            print(f'[{split} / {match_name} / {video_name}]\tvideo frames: {video_frame_count}')
            match_frame_count += video_frame_count
        get_match_median(match_dir)
        print(f'[{split} / {match_name}]:\ttotal frames: {match_frame_count}')
        split_frame_count += match_frame_count
    
    print(f'[{split}]:\ttotal frames: {split_frame_count}')
print('Done.')