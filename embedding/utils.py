import h5py
import os
from tqdm import tqdm
import yaml

import cv2
import numpy as np
import torch
import torch.nn.init
import torch.nn as nn

# resize image to size 32x32
cv2_scale36 = lambda x: cv2.resize(x, dsize=(36, 36),
                                 interpolation=cv2.INTER_LINEAR)
cv2_scale = lambda x: cv2.resize(x, dsize=(32, 32),
                                 interpolation=cv2.INTER_LINEAR)
# reshape image
np_reshape = lambda x: np.reshape(x, (32, 32, 3))

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x


def read_yaml(filename: str):
    with open(filename, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def read_h5(filename: str):
    dataset = {}
    with h5py.File(filename, 'r') as file:
        for k, _ in tqdm(file.items()):
            dataset[k] = np.array(file[k])
    return dataset

def to_3d_tensor(s: torch.tensor):
    s = s.view(64, 64, 1)
    return torch.cat((s, s, s), axis=2)

def write_h5(filename: str, descriptors: dict):
    if os.path.isfile(filename):
        raise RuntimeError(f"{filename} exists")
    h = h5py.File(filename, 'w')
    for key in descriptors.keys():
        h.create_dataset(key, data=descriptors[key])
    h.close()
