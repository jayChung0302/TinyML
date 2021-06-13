from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os, sys
import argparse
import logging
import time
from torchvision.transforms.transforms import RandomResizedCrop, RandomRotation, Resize
from tqdm import tqdm
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from theconf import ConfigArgumentParser
from theconf import Config as C
from utils import accuracy, save_checkpoint, create_exp_dir
from RandAugment import RandAugment
from model.pyramidnet import PyramidNet
from train import trainer

parser = ConfigArgumentParser(conflict_handler='resolve')
parser.add_argument('--data_dir', type=str, help='Dataset directory', default='/dataset')
parser.add_argument('--exp_name', type=str, default='dryrun', help='denote experiment name')
parser.add_argument('--checkpoint_path', type=str, default='checkpoint', help='checkpoint path')
parser.add_argument('--save_path', type=str, default='exp', help='checkpoint path')
parser.add_argument('--num_epoch', type=int, default=200, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
parser.add_argument('--num_classes', type=int, default=100, help='number of classes')
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--use_amp', action='store_true', help='using FP16')
parser.add_argument('--use_lars', action='store_true', help='using layer-wise adaptive rate scaling')

parser.add_argument('--log_cycle', type=float, default=0.1, help='tensorboard logging frequency')
# continue training
parser.add_argument('--is_continue', action='store_true', help='continue training')
parser.add_argument('--load_path', type=str, default=None, help='path for loading checkpoint')

#TODO:
# Continuing
# TinyTL
# progressive learning
# regularization turn off

args = parser.parse_args()
if args.use_amp:
    from apex import amp
if args.use_lars:
    from torchlars import LARS

args.exp_path = os.path.join(args.save_path, args.exp_name)
create_exp_dir(args.exp_path)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.exp_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():
    logging.info(f'Experiment date: {datetime.today().strftime("%Y-%m-%d-%H-%M")}')
    logging.info(vars(args))
    writer = SummaryWriter(f'logs/{args.exp_name}')
    if args.cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pin_memory = True
    else:
        device = torch.device("cpu")
        pin_memory = False

    head = nn.Linear(in_features=1696, out_features=args.num_classes)
    net = PyramidNet(dataset='imagenet', depth=101, alpha=360, num_classes=1000)
    wts = torch.load('./model/pyramidnet101_360.pth')
    net.load_state_dict(wts)
    net.fc = head
    logging.info(f'network: {net}')
    net = net.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, \
        weight_decay=args.weight_decay)
    if args.use_lars:
        optimizer = LARS(optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    
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
    # N, M=3, 13
    # train_transform.transforms.insert(0, RandAugment(N, M))
    data_transforms = {'train': train_transform, 'val': val_transform}
    logging.info(f'{net.__class__.__name__}')
    logging.info(f'{data_transforms}')
    logging.info(f'{optimizer}')
    logging.info(f'{scheduler.__class__.__name__}: {scheduler.state_dict()}')

    # image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x]) \
    #     for x in ['train', 'val']}
    image_datasets = {}
    image_datasets['train'] = datasets.CIFAR100(root='./dataset', train=True, download=True, transform=data_transforms['train'])
    image_datasets['val'] = datasets.CIFAR100(root='./dataset', train=False, download=True, transform=data_transforms['val'])

    if args.use_amp:
        # amp initialization
        net, optimizer = amp.initialize(net, optimizer, opt_level="O1")
    
    net, optimizer, scheduler = trainer(net, image_datasets, args.num_epoch, criterion, optimizer, scheduler, \
        device, pin_memory, args.exp_path, writer)

if __name__ == '__main__':
    main()
