import torch
from torch.nn.utils.weight_norm import weight_norm


def add_weight_normalization_layer(layer: torch.nn.Module, normalize: bool):
    if normalize:
        return weight_norm(layer, dim=None)
    return layer
