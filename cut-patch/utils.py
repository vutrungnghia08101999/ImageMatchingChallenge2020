from tqdm import tqdm
import h5py
import yaml


import torch
import torch.nn as nn
import numpy as np

def read_yaml(filename: str):
    with open(filename, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def read_h5(filename: str):
    data = {}
    with h5py.File(filename, 'r') as f:
        for key in tqdm(list(f.keys())):
            data[key] = np.array(f[key])
    return data


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x


def is_valid_patch(x: float, y: float, image: np.array, patch_size: int) -> bool:
    height = image.shape[0]
    width = image.shape[1]

    half_size = int(patch_size/2)
    if x - half_size < 0 or y - half_size < 0:
        return False
    if x - half_size + patch_size > width or y - half_size + patch_size > height:
        return False
    return True


def get_patch(x: float, y: float, image: np.array, patch_size: int) -> np.array:
    half_size = int(patch_size/2)
    
    x_start = int(x - half_size)
    y_start = int(y - half_size)
    assert x_start >= 0 and y_start >= 0
    assert x_start + patch_size <= image.shape[1] and y_start + patch_size <= image.shape[0]
    return image[y_start: y_start+patch_size, x_start: x_start+patch_size, :]
