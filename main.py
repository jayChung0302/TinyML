from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os, sys
import argparse
import logging
import time
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

from utils import accuracy, save_checkpoint, create_exp_dir
from RandAugment import RandAugment

parser = argparse.ArgumentParser(description='Regular training')
parser.add_argument('--data_dir', type=str, help='Dataset directory', default='/dataset')
parser.add_argument('--exp_name', type=str, default='dryrun', help='denote experiment name')
parser.add_argument('--checkpoint_path', type=str, default='checkpoint', help='checkpoint path')
parser.add_argument('--save_path', type=str, default='exp', help='checkpoint path')
parser.add_argument('--num_epoch', type=int, default=200, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
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
# Config
# progressive learning
# regularization turn off

args, _ = parser.parse_known_args()
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

    net = models.mobilenet_v2(pretrained=True)
    head = nn.Linear(in_features=1280, out_features=args.num_classes)
    net.classifier[1] = head
    net = net.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, \
        weight_decay=args.weight_decay)
    if args.use_lars:
        optimizer = LARS(optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
			transforms.Pad(4, padding_mode='reflect'),
			transforms.RandomCrop(32),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean,std=std)
		])
    val_transform = transforms.Compose([
			transforms.CenterCrop(32),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean,std=std)
		])
        
    train_transform.transforms.insert(0, RandAugment(N, M))
    data_transforms = {'train': train_transform, 'val': val_transform}
    logging.info(f'{net.__class__.__name__}')
    logging.info(f'{data_transforms}')
    logging.info(f'{optimizer}')
    logging.info(f'{scheduler.__class__.__name__}: {scheduler.state_dict()}')

    # image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x]) \
    #     for x in ['train', 'val']}
    image_datasets = {}
    image_datasets['train'] = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=data_transforms['train'])
    image_datasets['val'] = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=data_transforms['val'])

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, \
        shuffle=True, num_workers=4, pin_memory=pin_memory) \
        for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_iter = {x: dataset_sizes[x]//args.batch_size for x in ['train', 'val']}
    num_cycle = {x: int(num_iter[x] * args.log_cycle) for x in ['train', 'val']}
    logging.info(f'number of iter: {num_iter}')
    logging.info(f'iter cycle for tensorboard: {num_cycle}')
    logging.info('training start ......!')
    training_start_time = time.time()

    if args.use_amp:
        # amp initialization
        net, optimizer = amp.initialize(net, optimizer, opt_level="O1")
        
    best_acc = 0
    for epoch in range(1, args.num_epoch+1):
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
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        if args.use_amp:
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
                    loss_iters = running_loss/((i+1)*args.batch_size)
                    acc_iters = running_corrects/((i+1)*args.batch_size)
                    writer.add_scalar(f'{phase}/loss', loss_iters, idx)
                    writer.add_scalar(f'{phase}/accuracy', acc_iters, idx)
                    for name, weight in net.named_parameters():
                        writer.add_histogram(name, weight, epoch)
                        writer.add_histogram(f'{name}.grad', weight.grad, epoch)
                    logging.info(f'Acc: {acc_iters:.4f}, Loss: {loss_iters:.4f}, ({i}/{idx})')


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            logging.info(f'EPOCH:({epoch}/{args.num_epoch}) {phase} mode || Acc: {epoch_acc:.4f}, Loss: {epoch_loss:.4f}')

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
                
                if args.use_amp:
                    stats['amp_state_dict'] = amp.state_dict()

                save_checkpoint(stats, is_best, args.exp_path)

        epoch_duration = time.time() - epoch_start_time
        logging.info(f'epoch duration: {int(epoch_duration)}s')
    logging.info(f'End of training, whole session took {int(time.time() - training_start_time)}s')
    logging.info(f'Best validation accuracy: {best_acc}')

if __name__ == '__main__':
    main()
