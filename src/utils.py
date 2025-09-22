import os
import sys
import random

import torch
import torchvision
import numpy as np


def fix_random(seed: int) -> None:
    """
    Fixes the random seed for reproducibility.

    Args:
        seed (int): An integer representing the random seed.

    Returns:
        None
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # slower


def get_device() -> torch.device:
    """
    Returns the device to be used for tensor computations.

    Returns:
        torch.device: A device object representing the device to be used for tensor computations.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
    

def norm(tensor_map: torch.Tensor, clone: bool = True) -> torch.Tensor:
    """
    Normalizes a tensor map to the range [0, 1] along the channel dimension.

    Args:
        tensor_map (torch.Tensor): A tensor of shape (batch_size, channels, height, width) representing the tensor map.
        clone (bool): A boolean indicating whether to clone the input tensor before normalization (default: True).

    Returns:
        torch.Tensor: A tensor of shape (batch_size, channels, height, width) representing the normalized tensor map.
    """
    tensor_map_AA = tensor_map.clone() if clone else tensor_map
    tensor_map_AA = tensor_map_AA.view(tensor_map.size(0), -1)
    tensor_map_AA -= tensor_map_AA.min(1, keepdim=True)[0]
    tensor_map_AA /= (tensor_map_AA.max(1, keepdim=True)[0]-tensor_map_AA.min(1, keepdim=True)[0])
    tensor_map_AA = tensor_map_AA.view(tensor_map.shape)
    tensor_map_AA[torch.isnan(tensor_map_AA)]=0
    return tensor_map_AA

