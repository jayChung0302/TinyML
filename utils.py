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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
	
def computeTime(model, device='cuda'):
	from thop import profile
	from thop import clever_format
	import time
	inputs = torch.randn(1, 3, 224, 224) # input image 3ch for 3, 1ch for 1
	if device == 'cuda':
		model = model.cuda()
		inputs = inputs.cuda()

	model.eval()

	i = 0
	time_spent = []
	while i < 100:
		start_time = time.time()
		with torch.no_grad():
			out = model(inputs)

		if device == 'cuda':
			torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
		if i != 0:
			time_spent.append(time.time() - start_time)
		i += 1
	print('Avg execution time: {:.3f}'.format(np.mean(time_spent)))
	print(np.mean(time_spent))

if __name__ == '__main__':
	import torchvision.models as models
	net = models.resnet50()
	net.cuda()
	input = torch.randn(1, 3, 224, 224).cuda()
	flops, params = profile(net, inputs=(input, ))
	flops, params = clever_format([flops, params], "%.3f")
