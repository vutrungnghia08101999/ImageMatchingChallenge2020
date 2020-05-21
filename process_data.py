import argparse
import logging
import deepdish as dd
import h5py
import os
from time import time
from tqdm import tqdm
import time
import random
from PIL import Image
import pandas as pd
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from colmap.scripts.python.read_write_model import read_model, qvec2rotmat
from colmap.scripts.python.read_dense import read_array
from utils import is_valid_patch, get_patch, read_yaml, generate_test_set
from model import HardNet

# logging.basicConfig(filename='logs.txt',
#                     filemode='a',
#                     format='%(asctime)s, %(levelname)s: %(message)s',
#                     datefmt='%y-%m-%d %H:%M:%S',
#                     level=logging.DEBUG)
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# logging.getLogger().addHandler(console)
logging.basicConfig(level=logging.INFO)

np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='/media/vutrungnghia/New Volume/P2-ImageMatchingChallenge/dataset/challenge-dataset/valid')
parser.add_argument('--output', type=str, default='/media/vutrungnghia/New Volume/P2-ImageMatchingChallenge/dataset/superglue')
parser.add_argument('--model', type=str, default='/media/vutrungnghia/New Volume/P2-ImageMatchingChallenge/models/checkpoint_20-04-16_05-23-33_4_3100.pth')
parser.add_argument('--scene', type=str, default='reichstag')
args = parser.parse_args([])

INPUT = os.path.join(args.input, args.scene, 'dense')
OUTPUT = os.path.join(args.output, args.scene)
os.makedirs(OUTPUT, exist_ok=False)

# load rgb model
model = HardNet()
checkpoint = torch.load(args.model)
model.load_state_dict(checkpoint['state_dict'])

# load data from bin files
cameras, images, points = read_model(path=os.path.join(INPUT, 'sparse'), ext='.bin')

# load all data to RAM
images_storage = {}
image_files = list(os.listdir(os.path.join(INPUT, 'images')))
logging.info(f'Load {len(image_files)} images')
for filename in tqdm(image_files):
    images_storage[filename] = np.array(Image.open(os.path.join(INPUT, 'images', filename)))


# from collections import Counter
# matches = []
# for point_3d_id in tqdm(images[1].point3D_ids):
#     if point_3d_id == -1:
#         continue
#     point = points[point_3d_id]
#     matches = matches + list(point.image_ids)

# Counter(matches).keys()
# Counter(matches).values()

for image_id in tqdm(images.keys()):
    image_id = 1
    image = images[image_id]
    height, width, _ = images_storage[image.name].shape
    width_check = (image.xys[:, 0] - 16 > 0) * (image.xys[:, 0] + 16 < width)
    height_check = (image.xys[:, 1] - 16 > 0) * (image.xys[:, 1] + 16 < height)
    has_3d_points = image.point3D_ids != -1
    flags = width_check * height_check * has_3d_points
    image.keypoints = image.xys[flags]
    image.point3d_ids = image.point3D_ids[flags]
