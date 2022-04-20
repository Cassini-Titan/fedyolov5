import sys
import os
from logging import Logger
from typing import List, Optional
from pathlib import Path
from logging import Logger

import torch
import yaml
from torch.nn import Module

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.yolo import Model
from utils.general import LOGGER, check_dataset, intersect_dicts



def load_model(pt:str, opt, hyp:str)-> Model:
    # load hyperparameter
        # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    data_dict = check_dataset(opt.data)
    nc = int(data_dict['nc']) 
    anchors=hyp.get('anchors')
    # exclude = ['anchor'] if (hyp.get('anchors')) else []  # exclude keys
    # csd = ckpt['model'].float().state_dict()
    # csd = intersect_dicts(csd, model.state_dict(),
    #                           exclude=exclude)  # intersect
    # nc = 20 for voc
    ckpt = torch.load(pt, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
    model = Model(ckpt['model'].yaml, ch=3, nc=nc, anchors=anchors)
    exclude = ['anchor'] if hyp.get('anchors') else []
    model_info = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    weights = intersect_dicts(model_info, model.state_dict(), exclude=exclude)
    model.load_state_dict(weights,strict=False)
    del weights, ckpt
    return model

# Freeze layers for transfer learning
# freeze backbone:0-9,i.e. list(range(10))
# freeze non-output:0-23
def freeze_model(model:Module, freezed_layers:List):
    freeze = [f'model.{x}.' for x in (freezed_layers if len(freezed_layers) > 1 else range(freezed_layers[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False


