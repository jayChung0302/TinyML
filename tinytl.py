# https://arxiv.org/abs/2007.11622
import torchvision.models as models
import torch.nn as nn

__all__ = ['tinytlb', 'enable_bn_update', 'set_module_grad_status']


def tinytlb(net):
    set_module_grad_status(net, False)
    for layer in net.modules():
        for name, params in layer.named_parameters():
            if name == 'bias':
                params.requires_grad = True

def enable_bn_update(net):
    for module in net.modules():
        if type(module) in [nn.BatchNorm2d, nn.GroupNorm] and module.weight is not None:
            set_module_grad_status(module, True)

def set_module_grad_status(module, flag=False):
    if isinstance(module, list):
        for inner_module in module:
            set_module_grad_status(inner_module, flag)
    else:
        for param in module.parameters():
            param.requires_grad = flag


if __name__ == '__main__':
    net = models.resnet50()
    tinytlb(net)
    # enable_bn_update(net)
    for layer in net.modules():
        for name, params in layer.named_parameters():
            if params.requires_grad == True:
                print(name)
    
    