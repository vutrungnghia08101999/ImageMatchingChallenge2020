import os
import logging
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import random

from utils import to_3d_tensor


class SubmissionDataset(Dataset):
    def __init__(self, image: str, patches: dict, transform=None): 
        self.patches = patches[image]
        self.transform = transform

    def __getitem__(self, index):
        patch = self.patches[index]
        patch = to_3d_tensor(torch.tensor(patch))
        if self.transform is not None:
            patch = self.transform(patch.numpy())
        return patch


    def __len__(self):
        return self.patches.shape[0]
