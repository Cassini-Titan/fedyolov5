from logging import Logger
from typing import Optional
import warnings
import argparse
import sys
import os
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy

import math
import random
import time
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torch import device
from torch.nn import Module
from tqdm import tqdm
from tqdm.auto import tqdm

FILE = Path(__file__).resolve() #获取当前文件路径
print("*"*10)
print(FILE)
print("*"*20)
ROOT = FILE.parents[2]
print(ROOT)
print("1"*20)
print(sys.path)