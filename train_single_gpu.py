
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
import flwr as fl
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
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import val
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.general import (LOGGER, check_dataset, check_file, check_img_size,
                           check_suffix, check_yaml, colorstr, get_latest_run, increment_path, init_seeds,
                           intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods, one_cycle,
                           print_args, print_mutation, strip_optimizer)
from utils.loggers import Loggers
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_labels
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel
from utils.general import LOGGER, colorstr
from model_utils import load_model, freeze_model




def train(model:Model, hyp, opt, device, callbacks, loggers):  # hyp is path/to/hyp.yaml or hyp dictionary
    save_dir, epochs, batch_size, weights, data, cfg, noval, nosave, workers, freeze ,client_id= \
        opt.save_dir, opt.epochs, opt.batch_size, opt.weights, opt.data, opt.cfg, \
        opt.noval, opt.nosave, opt.workers, opt.freeze, opt.id
    callbacks.run('on_pretrain_routine_start')

    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    # LOGGER.info(colorstr('hyperparameters: ') +
    #             ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    # with open(save_dir / 'hyp.yaml', 'w') as f:
    #     yaml.safe_dump(hyp, f, sort_keys=False)
    # with open(save_dir / 'opt.yaml', 'w') as f:
    #     yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Loggers
    data_dict = None
    # loggers = Loggers(save_dir, weights, opt, hyp, logger)  # loggers instance



    # Config
    cuda = device.type != 'cpu'
    data_dict = data_dict or check_dataset(data)  # check if None
    train_path= Path(data_dict['train'][0]) /  f"client{client_id}"
    val_path = data_dict['val'][0]
    names = data_dict['names']
    nc = int(data_dict['nc'])  # number of classes
    assert len(
        names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check
    is_coco = isinstance(val_path, str) and val_path.endswith(
        'coco/val2017.txt')  # COCO dataset

    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    # if pretrained:
    #     # load checkpoint to CPU to avoid CUDA memory leak
    #     ckpt = torch.load(weights, map_location='cpu')
    # model_info = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get(
    #         'anchors'))  # create
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # gs = 32
    # else:
    #     model = Model(cfg, ch=3, nc=nc, anchors=hyp.get(
    #         'anchors')).to(device)  # create
    # if pretrained:
    #     model.to(device)
        # load checkpoint to CPU to avoid CUDA memory leak
        # ckpt = torch.load(weights, map_location='cpu')
        # model_info = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors'))  # create
    # else:
    #     model = Model(cfg, ch=3, nc=nc, anchors=hyp.get(
    #         'anchors')).to(device)  # create

    # Freeze
    freeze = [f'model.{x}.' for x in (freeze if len(
        freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            # LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    # Image size

    # verify imgsz is gs-multiple
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)

    # Optimizer
    nbs = 64  # nominal batch size
    # accumulate loss before optimizing
    accumulate = max(round(nbs / batch_size), 1)
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    g = [], [], []  # optimizer parameter groups
    bn = nn.BatchNorm2d, nn.LazyBatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d, nn.LazyInstanceNorm2d, nn.LayerNorm
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)

    optimizer = SGD(g[2], lr=hyp['lr0'],
                    momentum=hyp['momentum'], nesterov=True)
    # add g0 with weight_decay
    optimizer.add_param_group(
        {'params': g[0], 'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': g[1]})  # add g1 (BatchNorm2d weights)
    # LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
    #             f"{len(g[1])} weight (no decay), {len(g[0])} weight, {len(g[2])} bias")
    del g

    # Scheduler
    def lf(x): return (1 - x / epochs) * \
        (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    # plot_lr_scheduler(optimizer, scheduler, epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    # ema = ModelEMA(model)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    # if pretrained:
    #     # Optimizer
    #     if ckpt['optimizer'] is not None:
    #         optimizer.load_state_dict(ckpt['optimizer'])
    #         best_fitness = ckpt['best_fitness']

    #     # EMA
    #     # if ema and ckpt.get('ema'):
    #     #     ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
    #     #     ema.updates = ckpt['updates']

    #     # Epochs
    #     start_epoch = ckpt['epoch'] + 1
    #     if epochs < start_epoch:
    #         LOGGER.info(
    #             f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
    #         epochs += ckpt['epoch']  # finetune additional epochs

    #     del ckpt

    # Trainloader

    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size,
                                              image_weights=opt.image_weights,
                                              stride=gs,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              workers=workers,
                                              rect=opt.rect,
                                              prefix=colorstr('train: '),
                                              shuffle=True)
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    nb = len(train_loader)  # number of batches
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0 
    val_loader = create_dataloader(val_path,
                                   imgsz,
                                   batch_size,
                                   stride=gs,
                                   hyp=hyp,
                                   cache=None if noval else opt.cache,
                                   rect=opt.rect,
                                   workers=workers * 2,
                                   pad=0.5,
                                   prefix=colorstr('val: '))[0]

    labels = np.concatenate(dataset.labels, 0)
    plot_labels(labels, names, save_dir)
    # check_anchors(dataset, model=model_info, thr=hyp['anchor_t'], imgsz=imgsz)
    model.half().float()  # pre-reduce anchor precision

    callbacks.run('on_pretrain_routine_end')

    # Model attributes
    # number of detection layers (to scale hyps)
    nl = de_parallel(model).model[-1].nl
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(
        dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    # number of warmup iterations, max(3 epochs, 100 iterations)
    nw = max(round(hyp['warmup_epochs'] * nb), 100)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    results = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=opt.patience)
    compute_loss = ComputeLoss(model)  # init loss class
    callbacks.run('on_train_start')
    # LOGGER.info(f'Starting training for {epochs} epochs...')
    # epoch ------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        callbacks.run('on_train_epoch_start')
        model.train()

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        mloss = torch.zeros(3, device=device)  # mean losses
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 8) % ('Client', 'Epoch', 'gpu_mem',
                    'box', 'obj', 'cls', 'labels', 'img_size'))
        # progress bar
        pbar = tqdm(pbar, total=nb,
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        optimizer.zero_grad()
        # batch -------------------------------------------------------------
        for i, (imgs, targets, paths, _) in pbar:
            callbacks.run('on_train_batch_start')
            # number integrated batches (since train start)
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / \
                255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(
                    ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(
                        ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(
                            ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(
                    imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    # new shape (stretched to gs-multiple)
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = nn.functional.interpolate(
                        imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward 
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(
                    pred, targets.to(device))  # loss scaled by batch_size

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                # if ema:
                #     ema.update(model)
                last_opt_step = ni

            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            # (GB)
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            pbar.set_description(('%10i' + '%10s' * 2 + '%10.4g' * 5) %
                                 (client_id, f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
            callbacks.run('on_train_batch_end', ni, model, imgs,
                          targets, paths, True, False)
            if callbacks.stop_training:
                return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        # mAP
        callbacks.run('on_train_epoch_end', epoch=epoch)
        # ema.update_attr(
        #     model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
        final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
        if not noval or final_epoch:  # Calculate mAP
            results, maps, _ = val.run(client_id=client_id,
                                       data = data_dict,
                                       batch_size=batch_size,
                                       imgsz=imgsz,
                                       model=model,
                                       dataloader=val_loader,
                                       save_dir=save_dir,
                                       plots=True,
                                       callbacks=callbacks,
                                       compute_loss=compute_loss)

            # Update best mAP
            # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi = fitness(np.array(results).reshape(1, -1))
            if fi > best_fitness:
                best_fitness = fi
            # log_vals = list(mloss) + list(results) + lr
            # callbacks.run('on_fit_epoch_end', log_vals,
            #               epoch, best_fitness, fi)
            precision, recall, mAP50, mAP, loss = results[0],results[1],results[2], results[3], results[4]
            log_vals = [loss,precision,recall,mAP50,mAP]
            callbacks.run('on_val_end', log_vals)            

            # Save model
            if (not nosave) or (final_epoch):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    # 'ema': deepcopy(ema.ema).half(),
                    # 'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    # 'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                del ckpt
                callbacks.run('on_model_save', last, epoch,
                              final_epoch, best_fitness, fi)

            # Stop Single-GPU
            if stopper(epoch=epoch, fitness=fi):
                break

            # Stop DDP TODO: known issues shttps://github.com/ultralytics/yolov5/pull/4576
            # stop = stopper(epoch=epoch, fitness=fi)
            # if RANK == 0:
            #    dist.broadcast_object_list([stop], 0)  # broadcast 'stop' to all ranks

        # Stop DPP
        # with torch_distributed_zero_first(RANK):
        # if stop:
        #    break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    # LOGGER.info(
    #     f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) :.3f} s.')
    # for f in last, best:
    #     if f.exists():
    #         strip_optimizer(f)  # strip optimizers
    #         if f is best:
    #             LOGGER.info(f'\nValidating {f}...')
    #             results, _, _ = val.run(
    #                 client_id=client_id,
    #                 data=data_dict,
    #                 batch_size=batch_size,
    #                 imgsz=imgsz,
    #                 model=attempt_load(f, device).half(),
    #                 iou_thres=0.65 if is_coco else 0.60,  # best pycocotools results at 0.65
    #                 dataloader=val_loader,
    #                 save_dir=save_dir,
    #                 save_json=is_coco,
    #                 verbose=True,
    #                 plots=True,
    #                 callbacks=callbacks,
    #                 compute_loss=compute_loss)  # val best model with plots
    #             if is_coco:
    #                 callbacks.run('on_fit_epoch_end', list(
    #                     mloss) + list(results) + lr, epoch, best_fitness, fi)

    #     callbacks.run('on_train_end', last, best, True, epoch, results)
    #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    # torch.cuda.empty_cache()
    return results, dataset.n


