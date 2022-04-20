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




FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import val
from utils.loss import ComputeLoss
from utils.general import check_dataset, check_file, check_yaml, colorstr, increment_path
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from config import server_config
from models.yolo import Model
from model_utils import load_model, freeze_model


# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_per_process_memory_fraction(0.2)


config = server_config()
DEVICE = torch.device(f'cuda:{config.device}')
callbacks = Callbacks()

def eval(model: Model) -> Tuple[float, Dict[str, Scalar]]:


    model.to(DEVICE)
    batch_size = config.batch_size
    imgsz = config.imgsz
    hyp = config.hyp
    workers = config.workers
    data_dict = check_dataset(config.data)
    # 验证集路径
    val_path = data_dict['val']
    # 加载超参数
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    val_loader = create_dataloader(path=val_path,
                                imgsz=imgsz,
                                batch_size=batch_size,
                                stride=gs,
                                hyp=hyp,
                                cache=config.cache,
                                rect=True,
                                rank=-1,
                                workers=workers,
                                pad=0.5,
                                prefix=colorstr('server eval: '))[0]

    # load eval settings
    model.hyp=hyp
    results, _, _ = val.run(
        client_id=0,
        data=check_dataset(config.data),
        batch_size=batch_size,
        imgsz=config.imgsz,
        model=model,
        dataloader=val_loader,
        device=DEVICE,
        save_dir=config.save_dir,
        plots=False,
        callbacks=callbacks,
        compute_loss=ComputeLoss(model))

    precision, recall, mAP50, mAP, loss = results[0],results[1],results[2], results[3], results[4]
    # (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t

    return loss, {'precision': precision, 'recall': recall, 'mAP50':mAP50, 'mAP':mAP}


def evaluate_fn(weights: Weights) -> Tuple[float, Dict[str, Scalar]]:
    model = load_model(config.weights, config, config.hyp)
    model.set_weights(weights)
    loss, metrics = eval(model)
    return loss, metrics


def server_check(config):
    # Checks
    # print_args(vars(config))
    # # check_git_status()

    config.data, config.hyp, config.weights, config.project = \
        check_file(config.data), check_yaml(config.hyp), str(config.weights), str(config.project)  # checks
    config.save_dir = str(Path(config.project) / config.name/ f'server')

    # config.save_dir = str(increment_path(Path(config.project) / config.name
    #                                      / f'server' / 'round', exist_ok=config.exist_ok, mkdir=True))
    

if __name__ == "__main__":

    # Configure strategy of FL
    # initial global model
    # Client-side:train-> Dict[str,tensor] -> List[numpy] -> Parameters(List[byte])->output
    # Server-side:input->Parameters(List[byte])->List[numpy]->aggregate

    # check data and model
    server_check(config)

    # load and init global model 
    global_model = load_model(config.weights, config, config.hyp)
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

    # Start Flower server
    fl.server.start_server(
        server_address="[::]:1234",
        config={"num_rounds": config.round},
        strategy=strategy,
    )
