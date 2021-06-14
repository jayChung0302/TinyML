from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os, sys
import argparse
import logging
import time
from torchvision.transforms.transforms import RandomResizedCrop, RandomRotation, Resize
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
from utils import accuracy, save_checkpoint, create_exp_dir
from RandAugment import RandAugment
from model.pyramidnet import PyramidNet
from train import trainer

#TODO:
# Continuing - get model with config
# TinyTL
# progressive learning
# regularization turn off

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config")
def main(cfg:DictConfig) -> None:
    exp_path = os.path.join(cfg.exp.save_path, cfg.exp.exp_name)
    create_exp_dir(exp_path)
    if cfg.exp.use_amp:
        from apex import amp
    if cfg.exp.use_lars:
        from torchlars import LARS
    if cfg.exp.use_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pin_memory = True
    else:
        device = torch.device("cpu")
        pin_memory = False

    log.info(OmegaConf.to_yaml(cfg))
    log.info(f'Experiment date: {datetime.today().strftime("%Y-%m-%d-%H-%M")}')
    
    writer = SummaryWriter(f'logs/{cfg.exp.exp_name}')
    #TODO: get model
    head = nn.Linear(in_features=1696, out_features=cfg.dataset.num_classes)
    net = PyramidNet(dataset='imagenet', depth=101, alpha=360, num_classes=1000)
    wts = torch.load('./model/pyramidnet101_360.pth')
    net.load_state_dict(wts)
    net.fc = head
    log.info(f'network: {net}')
    net = net.to(device)
    #TODO: get loss_fn
    loss_fn = nn.CrossEntropyLoss().to(device)
    #TODO: get optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=cfg.params.lr, momentum=cfg.optmizer.momentum, \
        weight_decay=cfg.optimizer.weight_decay)
    if cfg.exp.use_lars:
        optimizer = LARS(optimizer)
    #TODO: get scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    #TODO: get dataset
    # mean = (0.4914, 0.4822, 0.4465)
    # std = (0.2470, 0.2435, 0.2616)
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)

    train_transform = transforms.Compose([
			# transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean,std=std)
		])
    val_transform = transforms.Compose([
			transforms.Resize(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean,std=std)
		])
    #TODO: RandAugment config
    # N, M=3, 13
    # train_transform.transforms.insert(0, RandAugment(N, M))
    data_transforms = {'train': train_transform, 'val': val_transform}
    log.info(f'{net.__class__.__name__}')
    log.info(f'{data_transforms}')
    log.info(f'{optimizer}')
    log.info(f'{scheduler.__class__.__name__}: {scheduler.state_dict()}')

    # image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x]) \
    #     for x in ['train', 'val']}
    image_datasets = {}
    image_datasets['train'] = datasets.CIFAR100(root='./dataset', train=True, download=True, transform=data_transforms['train'])
    image_datasets['val'] = datasets.CIFAR100(root='./dataset', train=False, download=True, transform=data_transforms['val'])

    if cfg.exp.use_amp:
        # amp initialization
        net, optimizer = amp.initialize(net, optimizer, opt_level="O1")
    
    net, optimizer, scheduler = trainer(net, image_datasets, cfg, loss_fn, optimizer, scheduler, \
        device, pin_memory, exp_path, writer)

if __name__ == '__main__':
    main()
