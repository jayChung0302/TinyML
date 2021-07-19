import os, sys
import omegaconf
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_optimizer(net, optcfg, lr, use_lars):
    if optcfg.name == "sgd":    
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=optcfg.momentum, \
            weight_decay=optcfg.weight_decay)

    if optcfg.name == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=optcfg.beta,\
             eps=optcfg.eps, weight_decay=optcfg.weight_decay, amsgrad=optcfg.amsgrad)
        
    if use_lars:
        from torchlars import LARS
        optimizer = LARS(optimizer)

    return optimizer

def get_scheduler(schedulecfg, optimizer):
    if schedulecfg.name == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=schedulecfg.T_max, \
            eta_min=schedulecfg.eta_min)
    
    if schedulecfg.name == 'step_lr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedulecfg.step_size, \
            gamma=schedulecfg.gamma, last_epoch=schedulecfg.last_epoch, verbose=schedulecfg.verbose)
    
    if schedulecfg.name == 'multi_step_lr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, schedulecfg.schedule)
        # (optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False)

    return scheduler

