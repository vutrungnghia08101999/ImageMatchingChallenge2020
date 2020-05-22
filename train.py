from pathlib import Path
import argparse
import random

import matplotlib.cm as cm
import numpy as np
import torch
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from dataset import SuperGlueDataset

from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)



torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='/media/vutrungnghia/New Volume/P2-ImageMatchingChallenge/dataset/superglue/output')
parser.add_argument('--train_scenes', type=str, default='reichstag,brandenburg_gate')
args = parser.parse_args([])


dataset = SuperGlueDataset(args.root, args.train_scenes.split(','))
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

s = iter(dataloader).next()

