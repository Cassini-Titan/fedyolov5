import argparse
import os
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def client_config(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--cfg', type=str,
                        default='', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='total batch size for all GPUs')
    parser.add_argument('--data', type=str, default=ROOT /
                        'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--device', type=int, default=0,
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights', type=str, default=ROOT /
                        'yolov5n.pt', help='initial weights path')
    parser.add_argument('--hyp', type=str, default=ROOT /
                        'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int,
                        default=640, help='train, val image size (pixels)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--freeze', nargs='+', type=int,
                        default=[0], help='Freeze layers: backbone=10, first3=0 1 2')

    parser.add_argument('--nosave', action='store_true',
                        help='only save final checkpoint')
    parser.add_argument('--cache', type=str, nargs='?', const='ram',
                        help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--noval', action='store_true',
                        help='only validate final epoch')
    parser.add_argument('--workers', type=int, default=0,
                        help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT /
                        'runs/train', help='save to project/name')
    parser.add_argument('--name', default='fedavg', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--label-smoothing', type=float,
                        default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100,
                        help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--multi-scale', action='store_true',
                        help='vary img-size +/- 50%%')

    return parser.parse_args()


def server_config(known=False):
    parser = argparse.ArgumentParser(description="Flower Server")
    parser.add_argument("--round", type=int, default=1, help='communication round')
    parser.add_argument('--data', type=str, default=ROOT /
                        'data/coco128.yaml', help='dataset.yaml path')  # public eval dataset
    parser.add_argument('--batch-size', type=int, default=8,
                        help='total batch size for all GPUs')
    parser.add_argument('--workers', type=int, default=0,
                        help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--hyp', type=str, default=ROOT /
                        'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int,
                        default=640, help='train, val image size (pixels)')
    parser.add_argument('--task', type=str, default='val', help='task')  
    parser.add_argument('--device', type=int, default=3,
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights', type=str, default=ROOT /
                        'yolov5n.pt', help='initial weights path')
    parser.add_argument('--project', default=ROOT /
                        'runs/train', help='save to project/name')
    parser.add_argument('--name', default='fedavg', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--cache', type=str, nargs='?', const='ram',
                        help='--cache images in "ram" (default) or "disk"')
    
    return parser.parse_args()



