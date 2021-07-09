from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def create_exp_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)
		print(f'Make dir: {path}')

def save_checkpoint(state, is_best, save_root):
	save_path = os.path.join(save_root, 'checkpoint.pth.tar')
	torch.save(state, save_path)
	if is_best:
		best_save_path = os.path.join(save_root, 'model_best.pth.tar')
		shutil.copyfile(save_path, best_save_path)

def load_checkpoint(load_root, is_best=False, filename='checkpoint.pth.tar'):
	if is_best:
		filename = 'model_best.pth.tar'
	load_path = os.path.join(load_root, filename)
	state = torch.load(load_path)
	return state

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred    = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

def kd_loss(y, labels, teacher_scores, T, alpha):
    # distll loss
    return nn.KLDivLoss()(F.log_softmax(y/T), \
		F.softmax(teacher_scores/T)) * (T*T * 2.0 + alpha) + F.cross_entropy(y,labels) * (1.-alpha)

