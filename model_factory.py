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

def get_config():
    print(locals())
    pass

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
            stats = load_checkpoint(cfg.exp.load_path, True)
            net = net()
            net.load_state_dict(stats['net_state_dict'])
            
    else:
        net = net()

    if datacfg.num_classes != 1000:
        #TODO: Add changing head feature
        # head = datacfg.dataset.num_classes
        pass


    
        
        
if __name__ == '__main__':
    get_config()
    getattr(model, 'pyramidnet')
    # get attribute
    net = getattr(models, 'resnet') # resnet
    getattr(net, 'resnet18')() # net
    getattr(models, 'resnet')
    