import sys
import os

import torch
import yaml
import flwr as fl
from typing import Tuple, Dict
from flwr.server.strategy import FedAvg
from flwr.common.parameter import weights_to_parameters
from flwr.common import Scalar, Weights
from pathlib import Path
from copy import deepcopy
from utils.loggers import Loggers




FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import val
from utils.loss import ComputeLoss
from utils.general import LOGGER, check_dataset, check_file, check_yaml, colorstr, methods
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from config import server_config
from models.yolo import Model
from model_utils import load_model, freeze_model
from utils.torch_utils import de_parallel
# loggers = Loggers(save_dir, CONFIG.weights, CONFIG, hyp, LOGGER)

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_per_process_memory_fraction(0.2)


CONFIG = server_config()
DEVICE = torch.device(f'cuda:{CONFIG.device}')
# MODEL = load_model(CONFIG.weights, CONFIG, CONFIG.hyp).to(DEVICE)
CALLBACKS = Callbacks()
# register logger
CONFIG.data, CONFIG.hyp, CONFIG.weights, CONFIG.project = \
    check_file(CONFIG.data), check_yaml(CONFIG.hyp), str(CONFIG.weights), str(CONFIG.project)  
CONFIG.save_dir = Path(CONFIG.project) / CONFIG.name / 'server'
loggers = Loggers(CONFIG.save_dir, CONFIG.weights, CONFIG, CONFIG.hyp, LOGGER)
for k in methods(loggers):
    CALLBACKS.register_action(k, callback=getattr(loggers, k))

def eval(model) -> Tuple[float, Dict[str, Scalar]]:

    model.to(DEVICE)
    ckpt = {
    'model': deepcopy(de_parallel(model)).half()}

    # Save last, best and delete
    torch.save(ckpt, CONFIG.save_dir / 'global.pt')
    # 验证集
    data_dict = check_dataset(CONFIG.data)
    val_path = data_dict['val']
    
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)

    val_loader = create_dataloader(path=val_path,
                                imgsz=CONFIG.imgsz,
                                batch_size=CONFIG.batch_size,
                                stride=gs,
                                hyp=model.hyp,
                                cache=CONFIG.cache,
                                rect=True,
                                rank=-1,
                                workers=CONFIG.workers,
                                pad=0.5,
                                prefix=colorstr('server eval: '))[0]

    # load eval settings
    results, _, _ = val.run(
        client_id=0,
        data=check_dataset(CONFIG.data),
        batch_size=CONFIG.batch_size,
        imgsz=CONFIG.imgsz,
        model=model,
        dataloader=val_loader,
        device=DEVICE,
        save_dir=CONFIG.save_dir,
        plots=False,
        callbacks=CALLBACKS,
        compute_loss=ComputeLoss(model),
        training=False)

    precision, recall, mAP50, mAP, loss = results[0],results[1],results[2], results[3], results[4]
    log_vals = [loss,precision,recall,mAP50,mAP]
    CALLBACKS.run('on_val_end', log_vals)
    # (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t

    return loss, {'precision': precision, 'recall': recall, 'mAP50':mAP50, 'mAP':mAP}


def evaluate_fn(weights: Weights) -> Tuple[float, Dict[str, Scalar]]:
    model = load_model(CONFIG.weights, CONFIG, CONFIG.hyp)
    model.set_weights(weights)
    loss, metrics = eval(model)
    return loss, metrics



    # CONFIG.save_dir = str(increment_path(Path(CONFIG.project) / CONFIG.name
    #
    #                                       / f'server' / 'round', exist_ok=CONFIG.exist_ok, mkdir=True))


if __name__ == "__main__":

    # Configure strategy of FL
    # initial global model
    # Client-side:train-> Dict[str,tensor] -> List[numpy] -> Parameters(List[byte])->output
    # Server-side:input->Parameters(List[byte])->List[numpy]->aggregate


    # load and init global model 
    global_model = load_model(CONFIG.weights, CONFIG, CONFIG.hyp)
    initial_parameters = weights_to_parameters(global_model.get_weights())
    del global_model
    
    strategy = FedAvg(
        fraction_fit=1,
        initial_parameters=initial_parameters,
        min_fit_clients=1,
        min_available_clients=1,
        min_eval_clients=1,
        eval_fn=evaluate_fn
    )
    msg_max_length = 1024 * 1024 * 1024
    # Start Flower server
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config={"num_rounds": CONFIG.round},
        strategy=strategy,
        grpc_max_message_length=msg_max_length
    )
