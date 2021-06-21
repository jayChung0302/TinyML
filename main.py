from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os, sys
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import hydra
from omegaconf import DictConfig, OmegaConf
from utils import accuracy, save_checkpoint, create_exp_dir, load_checkpoint
from RandAugment import RandAugment
from model.pyramidnet import PyramidNet
from train import trainer
from model_factory import get_model
from custom_dataset import get_dataset
from custom_optimizer import get_optimizer, get_scheduler
from tinytl import *

#TODO:
# Continuing - get model with config

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config")
def main(cfg:DictConfig) -> None:
    get_dataset(cfg.dataset)
    exp_path = os.path.join(cfg.exp.save_path, cfg.exp.exp_name)
    create_exp_dir(exp_path)
    if cfg.exp.use_amp:
        from apex import amp
    if cfg.exp.use_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pin_memory = True
    else:
        device = torch.device("cpu")
        pin_memory = False
    log.info(OmegaConf.to_yaml(cfg))
    writer = SummaryWriter(f'logs/{cfg.exp.exp_name}')

    net = get_model(cfg)

    head = nn.Linear(in_features=1696, out_features=cfg.dataset.num_classes)
    wts = torch.load('./model/pyramidnet101_360.pth')
    net.load_state_dict(wts)
    net.fc = head
    log.info(f'network: {net}')
    if cfg.exp.tiny_tl:
        tinytlb(net)
    net = net.to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = get_optimizer(net, cfg.optimizer, cfg.params.lr, cfg.exp.use_lars)
    scheduler = get_scheduler(cfg.scheduler, optimizer)
    
    #TODO: RandAugment config
    # N, M=3, 13
    # train_transform.transforms.insert(0, RandAugment(N, M))
    
    log.info(f'{net.__class__.__name__}')
    log.info(f'train transform: {cfg.dataset.train_trasnform}')
    log.info(f'val transform: {cfg.dataset.val_trasnform}')
    log.info(f'{optimizer}')
    log.info(f'{scheduler.__class__.__name__}: {scheduler.state_dict()}')
    
    image_datasets = get_dataset(datacfg=cfg.dataset)
    
    if cfg.exp.use_amp:
        # amp initialization
        net, optimizer = amp.initialize(net, optimizer, opt_level="O1")
    
    net, optimizer, scheduler = trainer(net, image_datasets, cfg, loss_fn, optimizer, scheduler, \
        device, pin_memory, exp_path, writer)

    if cfg.exp.reg_off:
        log.info(f'additional training with weaker or no regularization')
        image_datasets = get_dataset(datacfg=cfg.dataset)
        image_datasets['train'] = image_datasets['val']
        stats = load_checkpoint(exp_path, True)
        net.load_state_dict(stats['net_state_dict'])
        #TODO: weaker weight decay
        optimizer.load_state_dict(stats['optimizer_state_dict'])
        scheduler.load_state_dict(stats['scheduler_state_dict'])
        trainer(net, image_datasets, cfg, loss_fn, optimizer, scheduler, \
            device, pin_memory, exp_path, writer, extra=True)

if __name__ == '__main__':
    main()
