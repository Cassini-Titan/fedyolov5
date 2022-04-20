import argparse
from tqdm.auto import tqdm
from pathlib import Path

from exp.fedyolo.data.voc2yolo import convert_label

parser = argparse.ArgumentParser()
parser.add_argument('--num', type=int, default=4)
args = parser.parse_args()

# Convert
dir = Path('~/FL_Projects/exp/datasets/VOC/')
path = dir / f'VOCdevkit'
for year, image_set in ('2012', 'train'):
    for id in range(args.num):
        imgs_path = dir / 'images' / f'{image_set}{year}' / f'client{id+1}'
        lbs_path = dir / 'labels' / f'{image_set}{year}' / f'client{id+1}'
        imgs_path.mkdir(exist_ok=True, parents=True)
        lbs_path.mkdir(exist_ok=True, parents=True)
    
    # train 5717 val 5823

    with open(path / f'VOC{year}/ImageSets/Main/{image_set}.txt') as f:
        image_ids = f.read().strip().split()
    for id in tqdm(image_ids, desc=f'{image_set}{year}'):
        f = path / f'VOC{year}/JPEGImages/{id}.jpg'  # old img path
        lb_path = (lbs_path / f.name).with_suffix('.txt')  # new label path
        f.rename(imgs_path / f.name)  # move image
        convert_label(path, lb_path, year, id)  # convert labels to YOLO format