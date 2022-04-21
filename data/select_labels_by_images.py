from pathlib import Path
import os
import shutil
from tqdm import tqdm

is_train = False
set = 'val'
dir = Path('/home/zhang/FL_Projects/exp/datasets/')
imgs_path = dir / 'VOC' / 'images' / f'{set}2012'
lbs_old_path = dir / 'VOC_select' / 'labels' / f'{set}2012'
lbs_path = dir / 'VOC' / 'labels' / f'{set}2012'
client_num = 4

if is_train:
    for i in range(client_num):
        imgs_path_i = imgs_path / f'client{i+1}'
        lbs_path_i = lbs_path / f'client{i+1}'
        file_list = os.listdir(imgs_path_i)
        for img in tqdm(file_list):
            lb_name = Path(img).with_suffix('.txt')
            shutil.copyfile(lbs_old_path / lb_name, lbs_path_i/ lb_name)
else:
    file_list = os.listdir(imgs_path)
    for img in tqdm(file_list):
        lb_name = Path(img).with_suffix('.txt')
        shutil.copyfile(lbs_old_path / lb_name, lbs_path/ lb_name)


