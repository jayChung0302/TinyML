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
from model.pyramidnet import PyramidNet
from train import trainer
from model_factory import get_model
from custom_dataset import get_dataset
from custom_optimizer import get_optimizer, get_scheduler
from tinytl import *


log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config")
def main(cfg:DictConfig) -> None:
    exp_path = os.path.join(cfg.exp.save_path, cfg.exp.exp_name)
    if cfg.exp.use_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pin_memory = True
    else:
        device = torch.device("cpu")
        pin_memory = False
    log.info(OmegaConf.to_yaml(cfg))
    writer = SummaryWriter(f'logs/{cfg.exp.exp_name}')

    net = get_model(cfg)

    log.info(f'network: {net}')
    
    net = net.to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = get_optimizer(net, cfg.optimizer, cfg.params.lr, cfg.exp.use_lars)
    scheduler = get_scheduler(cfg.scheduler, optimizer)
    
    log.info(f'{net.__class__.__name__}')
    log.info(f'train transform: {cfg.dataset.train_transform}')
    log.info(f'val transform: {cfg.dataset.val_transform}')
    log.info(f'{optimizer}')
    log.info(f'{scheduler.__class__.__name__}: {scheduler.state_dict()}')
    
    image_datasets = get_dataset(datacfg=cfg.dataset)
    
    if cfg.exp.use_amp:
        net, optimizer = amp.initialize(net, optimizer, opt_level="O1")
    
    net, optimizer, scheduler = trainer(net, image_datasets, cfg, loss_fn, optimizer, scheduler, \
        device, pin_memory, exp_path, writer)

if __name__ == '__main__':
    main()
