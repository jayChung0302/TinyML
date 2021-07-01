from typing import Dict
import torch.nn as nn
import torchvision.models as models
from model import *
import model
from omegaconf import DictConfig, OmegaConf
from utils import load_checkpoint

AVAILABLE=[
    'mobilenet',
    'pyramidnet',
    'vit',
    'dino'
]
lib_keys = models.__dict__.keys()

def config_detail(net, modelcfg):
    if modelcfg.name == 'pyramidnet':
        return net(dataset=modelcfg.dataset, depth=modelcfg.depth, alpha=modelcfg.alpha, \
            num_classes=modelcfg.num_classes)
    elif modelcfg.name == 'SEResNext':
        pass
    else:
        return net()

def get_model(cfg: Dict) -> nn.Module:
    modelcfg, datacfg = cfg.model, cfg.dataset
    using_lib = False    
    if modelcfg.name in AVAILABLE:
        module = getattr(model, modelcfg.name)
    elif modelcfg.name in lib_keys:
        module = getattr(models, modelcfg.name)
        using_lib = True
    else:
        raise NotImplementedError(f'{modelcfg.name} is not implemented')

    net = getattr(module, modelcfg.version)

    if cfg.exp.is_continue:
        if using_lib:
            net = net(pretrained=True)
        else:
            net = config_detail(net, modelcfg)
            stats = load_checkpoint(cfg.exp.load_path, True)
            net.load_state_dict(stats['net_state_dict'])
            
    else:
        net = config_detail(net, modelcfg)
        
    if datacfg.num_classes != 1000:
        switch_head(net, datacfg.num_classes)

    return net

def switch_head(net, num_classes):
    head_name, head = list(net.named_children())[-1]
    if isinstance(head, nn.Sequential):
        head_name2, head2 = list(head.named_children())[-1]
        last_conv = getattr(getattr(net, head_name), head_name2)
        is_bias = True if last_conv.bias is not None else False
        setattr(getattr(net, head_name), head_name2, nn.Linear(last_conv.in_features, num_classes, bias=is_bias))
        return net
    else:
        last_conv = head
        is_bias = True if last_conv.bias is not None else False
        setattr(net, head_name, nn.Linear(last_conv.in_features, num_classes, bias=is_bias))
        return net
    #TODO: need to support fully conv layer / head may not be located on the last layer

if __name__ == '__main__':
    net1 = models.resnet50()
    net2 = models.vgg16()
    head_name, head = list(net2.named_children())[-1]
    
    import torch;
    inp = torch.randn(1,3,224,224)
    switch_head(net1, 10)
    print(net1(inp))
    import time; time.sleep(4)
    inp = torch.randn(1,3,224,224)
    switch_head(net2, 5)
    print(net2(inp))
