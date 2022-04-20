


from logging import Logger
from typing import Optional
import warnings
import argparse
import sys
import os
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy


import flwr as fl
import torch
from torch.utils.data import DataLoader
from torch.nn import Module


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val
from models.yolo import Model
from train_single_gpu import train
from utils.callbacks import Callbacks
from utils.general import LOGGER, check_file, check_yaml, increment_path
from model_utils import load_model, freeze_model
from config import client_config


# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=UserWarning)

# training config 
CONFIG = client_config()

# read args
DEVICE = torch.device(f'cuda:{CONFIG.device}')
CONFIG.data, CONFIG.cfg, CONFIG.hyp, CONFIG.weights, CONFIG.project = \
    check_file(CONFIG.data), check_yaml(CONFIG.cfg), check_yaml(
        CONFIG.hyp), str(CONFIG.weights), str(CONFIG.project)  # checks
CONFIG.save_dir = str(Path(CONFIG.project) / CONFIG.name/ f'client{CONFIG.id}')

# client device and local modol
MODEL = load_model(CONFIG.weights, CONFIG, CONFIG.hyp).to(DEVICE)
CALLBACKS = Callbacks()

class MobileClient(fl.client.NumPyClient):
    def __init__(self,id) -> None:
        self.id = id

    def get_parameters(self):
        params = [w.cpu().numpy() for w in MODEL.state_dict().values()]
        return params

    def set_parameters(self, parameters):
        params_dict = zip(MODEL.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        MODEL.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        # config：读取服务器对终端训练的配置信息
        self.set_parameters(parameters)
        local_train()
        data_length = 128
        return self.get_parameters(), data_length, {}

    def evaluate(self, parameters, config):
        # config：读取服务器对终端评估全局模型的配置信息
        self.set_parameters(parameters)
        loss, accuracy = 0.0, 0.0  # test
        data_length = 128
        return loss, data_length, {"accuracy": accuracy}


def local_train():
    # # Checks
    # print_args(vars(config))
    # # check_git_status()
    # Train
    train(model=MODEL, hyp=CONFIG.hyp, opt=CONFIG, device=DEVICE, callbacks=CALLBACKS)






if __name__ == '__main__':

    # 6 client on gpu 0,1,2
    # 1 server on gpu 3
    # --weights yolov5n.pt --batch-size 8 --workers 1 --noval --device client-id % torch.cuda.device_count-1
    # server --device torch.cuda.device_count-1
    # set training parameter
    
    # start local training
    # python client.py --data coco128.yaml --weights yolov5n.pt --img 640
    # python client.py --data coco128.yaml --weights yolov5n.pt --img 640 --freeze 10 --batch-size 8 --name fedavg --exist-ok
    # local_train(config)
        # Prepare model and dataset

    # Start Client
    fl.client.start_numpy_client(server_address="[::]:1234", client=MobileClient(CONFIG.id))
