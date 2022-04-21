import xml.etree.ElementTree as ET
import shutil
from tqdm.auto import tqdm
from pathlib import Path

# yaml = yaml.safe_load('./VOC.yaml')
# names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
#         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'] 
# index     0       1     2        3          4
names = ['person','car','bus','motorbike','bicycle'] # 选用种类 person, car, bus, bicycle
dir = Path('/home/zhang/FL_Projects/exp/datasets/VOC_select')
year = '2007'
set = 'trainval'
path = dir / f'VOCdevkit'
imgs_path = dir / 'images' / f'{set}{year}'
lbs_path = dir / 'labels' / f'{set}{year}'
imgs_path.mkdir(exist_ok=True, parents=True)
lbs_path.mkdir(exist_ok=True, parents=True)


def convert_label(path, lb_path, year, image_id):
    def convert_box(size, box):
        dw, dh = 1. / size[0], 1. / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh

    in_file = open(path / f'VOC{year}/Annotations/{image_id}.xml')
    out_file = open(lb_path, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls in names and not int(obj.find('difficult').text) == 1:
            xmlbox = obj.find('bndbox')
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
            cls_id = names.index(cls)  # class id
            out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')


# # Download
# dir = Path(yaml['path'])  # dataset root dir
# url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'
# urls = [url + 'VOCtrainval_06-Nov-2007.zip',  # 446MB, 5012 images
#         url + 'VOCtest_06-Nov-2007.zip',  # 438MB, 4953 images
#         url + 'VOCtrainval_11-May-2012.zip']  # 1.95GB, 17126 images
# download(urls, dir=dir / 'images', delete=False, curl=True, threads=3)
# Convert


for category in names:
    with open(path / f'VOC{year}/ImageSets/Main/{category}_{set}.txt') as f:
        for item in tqdm(f):
            id, has = item.strip().split()
            if has == '1':
                old_path = path / f'VOC{year}/JPEGImages/{id}.jpg'  # old img path
                lb_path = (lbs_path / old_path.name).with_suffix('.txt')  # new label path
                shutil.copyfile(old_path, imgs_path/ old_path.name)
                # old_path.rename(imgs_path / old_path.name)  # move image
                convert_label(path, lb_path, year, id)  # convert labels to YOLO format



            