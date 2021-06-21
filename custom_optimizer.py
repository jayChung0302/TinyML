import os, sys
import omegaconf
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_optimizer(net, cfg):
    if cfg.optimizer.name == "sgd":    
        optimizer = torch.optim.SGD(net.parameters(), lr=cfg.params.lr, momentum=cfg.optmizer.momentum, \
            weight_decay=cfg.optimizer.weight_decay)

    if cfg.optimizer.name == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.params.lr, betas=cfg.optimizer.beta,\
             eps=cfg.optimizer.eps, weight_decay=cfg.optimizer.weight_decay, amsgrad=cfg.optimizer.amsgrad)
        
    if cfg.exp.use_lars:
        from torchlars import LARS
        optimizer = LARS(optimizer)
    return optimizer
