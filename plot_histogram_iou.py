import argparse
import logging
import itertools
import os
from tqdm import tqdm
import random

import numpy as np
import seaborn as sns

from colmap.scripts.python.read_write_model import read_points3d_binary
logging.basicConfig(level=logging.INFO)

np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='/media/vutrungnghia/New Volume/P2-ImageMatchingChallenge/dataset/superglue/input')
parser.add_argument('--output', type=str, default='./histogram_iou')
parser.add_argument('--scene', type=str, default='brandenburg_gate')
args = parser.parse_args([])

logging.info('\n======== SUPERGLUE - DATA PREPROCESSING ========\n')
logging.info(args._get_kwargs())

INPUT = os.path.join(args.input, args.scene)
OUTPUT = os.path.join(args.output, f'{args.scene}.png')

# load data from bin files
points = read_points3d_binary(os.path.join(INPUT, 'dense/sparse/points3D.bin'))
local_features = np.load(os.path.join(INPUT, 'local_features.npy'), allow_pickle=True).item()

logging.info(f'No.points: {len(points)}')
logging.info(f'No.images: {len(local_features)}')
logging.info(f'max no.pairs: {int(len(local_features) * (len(local_features) - 1)/2)}')

# generate all pairs and list of matched index keypoints for each pairs
dataset = {}
for point_id in tqdm(points.keys()):
    point = points[point_id]
    if len(point.image_ids) < 2:
        continue
    pairs = itertools.combinations(sorted(point.image_ids), 2)
    pairs = list(pairs)
    for pair in pairs:
        if pair[0] == pair[1]:  # do not add pair of the same image
            continue
        if pair not in dataset:
            dataset[pair] = {'intersection': 0}
        dataset[pair]['intersection'] += 1

for u, v in dataset.keys():
    dataset[(u, v)]['union'] = local_features[u]['xys'].shape[0] + local_features[v]['xys'].shape[0] - dataset[(u, v)]['intersection']

iou = []
for k, v in dataset.items():
    iou.append(v['intersection']/v['union'])
histogram_iou = sns.distplot(iou)
file_out = histogram_iou.get_figure()
file_out.savefig(OUTPUT)
