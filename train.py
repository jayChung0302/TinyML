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

from utils import save_checkpoint
from RandAugment import RandAugment

def trainer(net, image_datasets, cfg, loss_fn, optimizer, scheduler, device, pin_memory, exp_path, writer):
    if cfg.exp.use_amp:
        from apex import amp
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_iter = {x: dataset_sizes[x] // cfg.params.batch_size for x in ['train', 'val']}
    num_cycle = {x: int(num_iter[x] * cfg.exp.log_cycle) for x in ['train', 'val']}
    logging.info(f'number of iter: {num_iter}')
    logging.info(f'iter cycle for tensorboard: {num_cycle}')
    logging.info('training start ......!')
    training_start_time = time.time()
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=cfg.params.batch_size, \
        shuffle=True, num_workers=4, pin_memory=pin_memory) \
        for x in ['train', 'val']}
    best_acc = 0
    for epoch in range(1, cfg.params.num_epoch+1):
        epoch_start_time = time.time()
        for phase in ['train', 'val']:
            running_loss = 0.0
            running_corrects = 0

            if phase == 'train':
                net.train()
            else:
                net.eval()

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_fn(outputs, labels)

                    if phase == 'train':
                        if cfg.exp.use_amp:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else: loss.backward()
                        optimizer.step()
                        scheduler.step()

                # loss is already divided by the batch size
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if i % num_cycle[phase] == num_cycle[phase]-1:
                    idx = (epoch-1) * num_iter[phase] + i
                    loss_iters = running_loss/((i+1)*cfg.params.batch_size)
                    acc_iters = running_corrects/((i+1)*cfg.params.batch_size)
                    writer.add_scalar(f'{phase}/loss', loss_iters, idx)
                    writer.add_scalar(f'{phase}/accuracy', acc_iters, idx)
                    for name, weight in net.named_parameters():
                        writer.add_histogram(name, weight, epoch)
                        writer.add_histogram(f'{name}.grad', weight.grad, epoch)
                    logging.info(f'Acc: {acc_iters:.4f}, Loss: {loss_iters:.4f}, ({i}/{idx})')


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            logging.info(f'EPOCH:({epoch}/{cfg.params.num_epochnum_epoch}) {phase} mode || Acc: {epoch_acc:.4f}, Loss: {epoch_loss:.4f} || current best Acc: {best_acc:.4f}')

            if phase == 'val':
                is_best = False
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    is_best = True
                logging.info('Saving models......')
                stats = {'last_epoch': epoch,
                        'acc': epoch_acc,
                        'loss': epoch_loss,
                        'net_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),}
                
                if cfg.exp.use_amp:
                    stats['amp_state_dict'] = amp.state_dict()

                save_checkpoint(stats, is_best, exp_path)

        epoch_duration = time.time() - epoch_start_time
        logging.info(f'epoch duration: {int(epoch_duration)}s')
    logging.info(f'End of training, whole session took {int(time.time() - training_start_time)}s')
    logging.info(f'Best validation accuracy: {best_acc}')
    
    return net, optimizer, scheduler
