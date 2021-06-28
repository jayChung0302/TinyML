# https://arxiv.org/abs/2007.11622
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict

__all__ = ['tinytlb', 'enable_bn_update', 'set_module_grad_status']


def tinytlb(net):
    """freeze weight, set bias trainable"""
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

def set_bn_param(net, momentum, eps, gn_channel_per_group=None, ws_eps=None, **kwargs):
    replace_bn_with_gn(net, gn_channel_per_group)

    for m in net.modules():
        if type(m) in [nn.BatchNorm1d, nn.BatchNorm2d]:
            m.momentum = momentum
            m.eps = eps
        elif isinstance(m, nn.GroupNorm):
            m.eps = eps

    replace_conv2d_with_Custom_conv2d(net, ws_eps)
    return


def get_bn_param(net):
    ws_eps = None
    for m in net.modules():
        if isinstance(m, CustomConv2d):
            ws_eps = m.WS_EPS
            break
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            return {
                'momentum': m.momentum,
                'eps': m.eps,
                'ws_eps': ws_eps,
            }
        elif isinstance(m, nn.GroupNorm):
            return {
                'momentum': None,
                'eps': m.eps,
                'gn_channel_per_group': m.num_channels // m.num_groups,
                'ws_eps': ws_eps,
            }
    return None

def min_divisible_value(n1, v1):
	""" make sure v1 is divisible by n1, otherwise decrease v1 """
	if v1 >= n1:
		return n1
	while n1 % v1 != 0:
		v1 -= 1
	return v1

def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def replace_bn_with_gn(model, gn_channel_per_group):
    if gn_channel_per_group is None:
        return

    for m in model.modules():
        to_replace_dict = {}
        for name, sub_m in m.named_children():
            if isinstance(sub_m, nn.BatchNorm2d):
                num_groups = sub_m.num_features // min_divisible_value(sub_m.num_features, gn_channel_per_group)
                gn_m = nn.GroupNorm(num_groups=num_groups, num_channels=sub_m.num_features, eps=sub_m.eps, affine=True)

                # load weight
                gn_m.weight.data.copy_(sub_m.weight.data)
                gn_m.bias.data.copy_(sub_m.bias.data)
                # load requires_grad
                gn_m.weight.requires_grad = sub_m.weight.requires_grad
                gn_m.bias.requires_grad = sub_m.bias.requires_grad

                to_replace_dict[name] = gn_m
        m._modules.update(to_replace_dict)


def replace_conv2d_with_Custom_conv2d(net, ws_eps=None):
    if ws_eps is None:
        return

    for m in net.modules():
        to_update_dict = {}
        for name, sub_module in m.named_children():
            if isinstance(sub_module, nn.Conv2d) and not sub_module.bias:
                # only replace conv2d layers that are followed by normalization layers (i.e., no bias)
                to_update_dict[name] = sub_module
        for name, sub_module in to_update_dict.items():
            m._modules[name] = CustomConv2d(
                sub_module.in_channels, sub_module.out_channels, sub_module.kernel_size, sub_module.stride,
                sub_module.padding, sub_module.dilation, sub_module.groups, sub_module.bias,
            )
            # load weight
            m._modules[name].load_state_dict(sub_module.state_dict())
            # load requires_grad
            m._modules[name].weight.requires_grad = sub_module.weight.requires_grad
            if sub_module.bias is not None:
                m._modules[name].bias.requires_grad = sub_module.bias.requires_grad
    # set ws_eps
    for m in net.modules():
        if isinstance(m, CustomConv2d):
            m.WS_EPS = ws_eps


def init_models(net, model_init='he_fout'):
    """
        Conv2d,
        BatchNorm2d, BatchNorm1d, GroupNorm
        Linear,
    """
    if isinstance(net, list):
        for sub_net in net:
            init_models(sub_net, model_init)
        return
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            if model_init == 'he_fout':
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif model_init == 'he_fin':
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            else:
                raise NotImplementedError
            if m.bias is not None:
                m.bias.data.zero_()
        elif type(m) in [nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm]:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.zero_()

class CustomModule(nn.Module):
    def forward(self, x):
        raise NotImplementedError
    
    @property
    def module_str(self,):
        raise NotImplementedError
    
    @property
    def config(self):
        raise NotImplementedError
    
    @staticmethod
    def build_from_config(self):
        raise NotImplementedError

class CustomNetwork(CustomModule):
    CHANNEL_DIVISIBLE = 8

    def forward(self, x):
        raise NotImplementedError

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def zero_last_gamma(self):
        raise NotImplementedError

    @property
    def grouped_block_index(self):
        raise NotImplementedError

    """ implemented methods """

    def set_bn_param(self, momentum, eps, gn_channel_per_group=None, **kwargs):
        set_bn_param(self, momentum, eps, gn_channel_per_group, **kwargs)

    def get_bn_param(self):
        return get_bn_param(self)

    def get_parameters(self, keys=None, mode='include'):
        if keys is None:
            for name, param in self.named_parameters():
                if param.requires_grad: yield param
        elif mode == 'include':
            for name, param in self.named_parameters():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag and param.requires_grad: yield param
        elif mode == 'exclude':
            for name, param in self.named_parameters():
                flag = True
                for key in keys:
                    if key in name:
                        flag = False
                        break
                if flag and param.requires_grad: yield param
        else:
            raise ValueError('do not support: %s' % mode)

class CustomConv2d(nn.Conv2d):
    """
    Conv2d with Weight Standardization
    https://github.com/joe-siyuan-qiao/WeightStandardization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(CustomConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.WS_EPS = None

    def weight_standardization(self, weight):
        if self.WS_EPS is not None:
            weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            weight = weight - weight_mean
            std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + self.WS_EPS
            weight = weight / std.expand_as(weight)
        return weight

    def forward(self, x):
        if self.WS_EPS is None:
            return super(CustomConv2d, self).forward(x)
        else:
            return F.conv2d(x, self.weight_standardization(self.weight), self.bias,
                            self.stride, self.padding, self.dilation, self.groups)

    def __repr__(self):
        return super(CustomConv2d, self).__repr__()[:-1] + ', ws_eps=%s)' % self.WS_EPS

def set_layer_from_config(layer_config):
	if layer_config is None:
		return None
	name2layer = {
		LiteResidualModule.__name__: LiteResidualModule,
		# ReducedMBConvLayer.__name__: ReducedMBConvLayer,
	}
	layer_name = layer_config.pop('name')
	if layer_name in name2layer:
		layer = name2layer[layer_name]
		return layer.build_from_config(layer_config)
	else:
		return ({'name': layer_name, **layer_config})

class Custom2DLayer(CustomModule):

    def __init__(self, in_channels, out_channels,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
        super(Custom2DLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ modules """
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules['bn'] = nn.BatchNorm2d(in_channels)
            else:
                modules['bn'] = nn.BatchNorm2d(out_channels)
        else:
            modules['bn'] = None
        # activation
        modules['act'] = build_activation(self.act_func, self.ops_list[0] != 'act')
        # dropout
        if self.dropout_rate > 0:
            modules['dropout'] = nn.Dropout2d(self.dropout_rate, inplace=True)
        else:
            modules['dropout'] = None
        # weight
        modules['weight'] = self.weight_op()

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == 'weight':
                # dropout before weight operation
                if modules['dropout'] is not None:
                    self.add_module('dropout', modules['dropout'])
                for key in modules['weight']:
                    self.add_module(key, modules['weight'][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def weight_op(self):
        raise NotImplementedError

    def forward(self, x):
        # similar to nn.Sequential
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError
class LiteResidualModule(CustomModule):

	def __init__(self, main_branch, in_channels, out_channels,
	             expand=1.0, kernel_size=3, act_func='relu', n_groups=2,
	             downsample_ratio=2, upsample_type='bilinear', stride=1):
		super(LiteResidualModule, self).__init__()

		self.main_branch = main_branch

		self.lite_residual_config = {
			'in_channels': in_channels,
			'out_channels': out_channels,
			'expand': expand,
			'kernel_size': kernel_size,
			'act_func': act_func,
			'n_groups': n_groups,
			'downsample_ratio': downsample_ratio,
			'upsample_type': upsample_type,
			'stride': stride,
		}

		kernel_size = 1 if downsample_ratio is None else kernel_size

		padding = get_same_padding(kernel_size)
		if downsample_ratio is None:
			pooling = CustomGlobalAvgPool2d()
		else:
			pooling = nn.AvgPool2d(downsample_ratio, downsample_ratio, 0)
		num_mid = make_divisible(int(in_channels * expand), divisor=CustomNetwork.CHANNEL_DIVISIBLE)
		self.lite_residual = nn.Sequential(OrderedDict({
			'pooling': pooling,
			'conv1': nn.Conv2d(in_channels, num_mid, kernel_size, stride, padding, groups=n_groups, bias=False),
			'bn1': nn.BatchNorm2d(num_mid),
			'act': build_activation(act_func),
			'conv2': nn.Conv2d(num_mid, out_channels, 1, 1, 0, bias=False),
			'final_bn': nn.BatchNorm2d(out_channels),
		}))

		# initialize
		init_models(self.lite_residual)
		self.lite_residual.final_bn.weight.data.zero_()

	def forward(self, x):
		main_x = self.main_branch(x)
		lite_residual_x = self.lite_residual(x)
		if self.lite_residual_config['downsample_ratio'] is not None:
			lite_residual_x = F.upsample(lite_residual_x, main_x.shape[2:],
			                             mode=self.lite_residual_config['upsample_type'])
		return main_x + lite_residual_x

	@property
	def module_str(self):
		return self.main_branch.module_str + ' + LiteResidual(downsample=%s, n_groups=%s, expand=%s, ks=%s)' % (
			self.lite_residual_config['downsample_ratio'], self.lite_residual_config['n_groups'],
			self.lite_residual_config['expand'], self.lite_residual_config['kernel_size'],
		)

	@property
	def config(self):
		return {
			'name': LiteResidualModule.__name__,
			'main': self.main_branch.config,
			'lite_residual': self.lite_residual_config,
		}

	@staticmethod
	def build_from_config(config):
		main_branch = set_layer_from_config(config['main'])
		lite_residual_module = LiteResidualModule(
			main_branch, **config['lite_residual']
		)
		return lite_residual_module

	def __repr__(self):
		return '{\n (main branch): ' + self.main_branch.__repr__() + ', ' + \
		       '\n (lite residual): ' + self.lite_residual.__repr__() + '}'

	@staticmethod
	def insert_lite_residual(net, downsample_ratio=2, upsample_type='bilinear',
	                         expand=1.0, max_kernel_size=5, act_func='relu', n_groups=2,
	                         **kwargs):
		if LiteResidualModule.has_lite_residual_module(net):
			# skip if already has lite residual modules
			return
		from ofa.imagenet_classification.networks import ProxylessNASNets
		if isinstance(net, ProxylessNASNets):
			bn_param = net.get_bn_param()

			# blocks
			max_resolution = 128
			stride_stages = [2, 2, 2, 1, 2, 1]
			for block_index_list, stride in zip(net.grouped_block_index, stride_stages):
				for i, idx in enumerate(block_index_list):
					block = net.blocks[idx].conv
					if isinstance(block, ZeroLayer):
						continue
					s = stride if i == 0 else 1
					block_downsample_ratio = downsample_ratio
					block_resolution = max(1, max_resolution // block_downsample_ratio)
					max_resolution //= s

					kernel_size = max_kernel_size
					if block_resolution == 1:
						kernel_size = 1
						block_downsample_ratio = None
					else:
						while block_resolution < kernel_size:
							kernel_size -= 2
					net.blocks[idx].conv = LiteResidualModule(
						block, block.in_channels, block.out_channels, expand=expand, kernel_size=kernel_size,
						act_func=act_func, n_groups=n_groups, downsample_ratio=block_downsample_ratio,
						upsample_type=upsample_type, stride=s,
					)

			net.set_bn_param(**bn_param)
		else:
			raise NotImplementedError

	@staticmethod
	def has_lite_residual_module(net):
		for m in net.modules():
			if isinstance(m, LiteResidualModule):
				return True
		return False

	@property
	def in_channels(self):
		return self.lite_residual_config['in_channels']

	@property
	def out_channels(self):
		return self.lite_residual_config['out_channels']

class CustomGlobalAvgPool2d(nn.Module):

    def __init__(self, keep_dim=True):
        super(CustomGlobalAvgPool2d, self).__init__()
        self.keep_dim = keep_dim

    def forward(self, x):
        return x.mean(3, keepdim=self.keep_dim).mean(2, keepdim=self.keep_dim)

    def __repr__(self):
        return 'CustomGlobalAvgPool2d(keep_dim=%s)' % self.keep_dim

class ZeroLayer(CustomModule):

    def __init__(self, stride):
        super(ZeroLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        raise ValueError

    @property
    def module_str(self):
        return 'Zero'

    @property
    def config(self):
        return {
            'name': ZeroLayer.__name__,
            'stride': self.stride,
        }

    @staticmethod
    def build_from_config(config):
        return ZeroLayer(**config)

class IdentityLayer(Custom2DLayer):

    def __init__(self, in_channels, out_channels,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        super(IdentityLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        return None

    @property
    def module_str(self):
        return 'Identity'

    @property
    def config(self):
        return {
            'name': IdentityLayer.__name__,
            **super(IdentityLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return IdentityLayer(**config)

def build_activation(act_func, inplace=True):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func == 'h_swish':
        raise ValueError('do not support: %s' % act_func)
        # return Hswish(inplace=inplace)
    elif act_func == 'h_sigmoid':
        raise ValueError('do not support: %s' % act_func)
        # return Hsigmoid(inplace=inplace)
    elif act_func is None or act_func == 'none':
        return None
    else:
        raise ValueError('do not support: %s' % act_func)

def get_same_padding(kernel_size):
	if isinstance(kernel_size, tuple):
		assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
		p1 = get_same_padding(kernel_size[0])
		p2 = get_same_padding(kernel_size[1])
		return p1, p2
	assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
	assert kernel_size % 2 > 0, 'kernel size should be odd number'
	return kernel_size // 2

class ProxylessNASNets(CustomNetwork):

    def __init__(self, first_conv, blocks, feature_mix_layer, classifier):
        super(ProxylessNASNets, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        if self.feature_mix_layer is not None:
            x = self.feature_mix_layer(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        for block in self.blocks:
            _str += block.module_str + '\n'
        _str += self.feature_mix_layer.module_str + '\n'
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            'name': ProxylessNASNets.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'feature_mix_layer': None if self.feature_mix_layer is None else self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config['first_conv'])
        feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
        classifier = set_layer_from_config(config['classifier'])

        blocks = []
        for block_config in config['blocks']:
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))

        net = ProxylessNASNets(first_conv, blocks, feature_mix_layer, classifier)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, MobileInvertedResidualBlock):
                if isinstance(m.mobile_inverted_conv, MBInvertedConvLayer) and isinstance(m.shortcut, IdentityLayer):
                    m.mobile_inverted_conv.point_linear.bn.weight.data.zero_()

if __name__ == '__main__':
    net = models.resnet50()
    tinytlb(net)
    # enable_bn_update(net)
    for layer in net.modules():
        for name, params in layer.named_parameters():
            if params.requires_grad == True:
                print(name)
    
    