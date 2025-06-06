import torch
import torch_pruning as tp
from typing import Sequence

from ..modules.conv import RepConv
class RepConvPruner(tp.pruner.BasePruningFunc):
    TARGET_MODULES = RepConv
    def prune_out_channels(self, layer: RepConv, idxs: Sequence[int]):
        layer.c2 = layer.c2 - len(idxs)
        
        tp.prune_conv_out_channels(layer.conv1.conv, idxs)
        tp.prune_conv_out_channels(layer.conv2.conv, idxs)
        tp.prune_batchnorm_out_channels(layer.conv1.bn, idxs)
        tp.prune_batchnorm_out_channels(layer.conv2.bn, idxs)
        return layer
    
    def prune_in_channels(self, layer: RepConv, idxs: Sequence[int]):
        layer.c1 = layer.c1 - len(idxs)
        
        tp.prune_conv_in_channels(layer.conv1.conv, idxs)
        tp.prune_conv_in_channels(layer.conv2.conv, idxs)
        return layer

    def get_out_channels(self, layer: RepConv):
        return layer.c2

    def get_in_channels(self, layer: RepConv):
        return layer.c1

    def get_in_channel_groups(self, layer: RepConv):
        return layer.g
    
    def get_out_channel_groups(self, layer: RepConv):
        return layer.g

from ..backbone.convnextv2 import LayerNorm
class LayerNormPruner(tp.pruner.BasePruningFunc):
    def prune_out_channels(self, layer:LayerNorm, idxs: Sequence[int]):
        num_features = layer.normalized_shape[0]
        keep_idxs = torch.tensor(list(set(range(num_features)) - set(idxs)))
        keep_idxs.sort()
        layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, -1)
        layer.bias = self._prune_parameter_and_grad(layer.bias, keep_idxs, -1)
        layer.normalized_shape = (len(keep_idxs),)
    
    prune_in_channels = prune_out_channels
    
    def get_out_channels(self, layer):
        return layer.normalized_shape[0]

    def get_in_channels(self, layer):
        return layer.normalized_shape[0]

from .block import RepConvN
class RepConvNPruner(tp.pruner.BasePruningFunc):
    def prune_out_channels(self, layer: RepConvN, idxs: Sequence[int]):
        layer.c2 = layer.c2 - len(idxs)
        
        tp.prune_conv_out_channels(layer.conv1.conv, idxs)
        tp.prune_conv_out_channels(layer.conv2.conv, idxs)
        tp.prune_batchnorm_out_channels(layer.conv1.bn, idxs)
        tp.prune_batchnorm_out_channels(layer.conv2.bn, idxs)
        return layer
    
    def prune_in_channels(self, layer: RepConvN, idxs: Sequence[int]):
        layer.c1 = layer.c1 - len(idxs)
        
        tp.prune_conv_in_channels(layer.conv1.conv, idxs)
        tp.prune_conv_in_channels(layer.conv2.conv, idxs)
        return layer

    def get_out_channels(self, layer: RepConvN):
        return layer.c2

    def get_in_channels(self, layer: RepConvN):
        return layer.c1

    def get_in_channel_groups(self, layer: RepConvN):
        return layer.g
    
    def get_out_channel_groups(self, layer: RepConvN):
        return layer.g