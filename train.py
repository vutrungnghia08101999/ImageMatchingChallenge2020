import argparse
import logging
import random

import matplotlib.cm as cm
import numpy as np
import torch
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader

from dataset import SuperGlueDataset
from model import SuperGlue
from utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, AverageMeter)

logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
# logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str,
                    default='/media/vutrungnghia/New Volume/P2-ImageMatchingChallenge/dataset/superglue/output')
parser.add_argument('--train_scenes', type=str,
                    default='reichstag')
parser.add_argument('--sinkhorn_iterations', type=int, default=20,
                    help='Number of Sinkhorn iterations performed by SuperGlue')
parser.add_argument('--match_threshold', type=float, default=0.2,
                    help='SuperGlue match threshold')
parser.add_argument('--weights', type=str,
                    default='')
parser.add_argument('--n_epochs', type=int, default=50)
args = parser.parse_args()
logging.info(f'\n========= TRIANING ==========\n')
logging.info(args._get_kwargs())

# creat dataset and dataloader
dataset = SuperGlueDataset(args.root, args.train_scenes.split(','))
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# create superglue model and load checkpoint if exist
configs = {
    'sinkhorn_iterations': args.sinkhorn_iterations,
    'match_threshold': args.match_threshold}

model = SuperGlue(configs)
if torch.cuda.is_available():
    model = model.cuda()
if args.weights:
    checkpoint = torch.load(args.weights)
    model.load_state_dict(checkpoint)

# create optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.001,
                             betas=(0.9, 0.999))

# training
for epoch in range(args.n_epochs):
    logging.info(f'EPOCH: {epoch+1}/{args.n_epochs}')
    model.train()
    avg_loss = AverageMeter()
    for pair in dataloader:
        # pair = iter(dataloader).next()  # dev
        groundtruth = pair['groundtruth']
        inputs = {
            'shape0': pair['shape'][0],
            'descriptors0': pair['descriptors'][0],
            'keypoints0': pair['keypoints'][0],
            'scores0': pair['scores'][0],
            'shape1': pair['shape'][1],
            'descriptors1': pair['descriptors'][1],
            'keypoints1': pair['keypoints'][1],
            'scores1': pair['scores'][1]
        }

        log_matrix = model(inputs)  # log_matrix.exp() satifys: sum(row) = sum(col) = 1
        if torch.cuda.is_available():
            loss = -1.0 * (log_matrix * groundtruth.type(torch.cuda.FloatTensor)).sum()
        else:
            loss = -1.0 * (log_matrix * groundtruth.type(torch.FloatTensor)).sum()
        avg_loss.update(loss.item()/groundtruth.sum().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(avg_loss.avg)
