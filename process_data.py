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

from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
import torch

from colmap.scripts.python.read_write_model import read_model, qvec2rotmat
from colmap.scripts.python.read_dense import read_array
from utils import is_valid_patch, get_patch, read_yaml, generate_test_set

logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

np.random.seed(0)
random.seed(0)

############### PARAMETERS ##############
configs = read_yaml('configs.yml')
parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str)
args = parser.parse_args()

configs['scene'] = args.scene
INPUT = os.path.join(configs['input'], configs['scene'], 'dense')
OUTPUT = os.path.join(configs['output'], configs['scene'])
#########################################
if not os.path.isdir(INPUT):
    raise RuntimeError(f'{INPUT} does not exist!')
os.makedirs(OUTPUT, exist_ok=False)

###########################################################################
"""
Go throught list of 3d points and cut exactly 2 patches and store to dataset
"""

cameras, images, points = read_model(path=os.path.join(INPUT, 'sparse'), ext='.bin')

images_storage = {}
image_files = list(os.listdir(os.path.join(INPUT, 'images')))
logging.info(f'Load {len(image_files)} images')
for filename in tqdm(image_files):
    images_storage[filename] = plt.imread(os.path.join(INPUT, 'images', filename))

rgb = {}
gray = {}
pbar = list(points.keys())
n_3d_points = len(pbar)
pbar.sort()
logging.info(f'Get 2 patches per 3d point of {len(pbar)} 3d points')
for point_id in tqdm(pbar):
    point = points[point_id]
    image_ids = point.image_ids
    point2D_idxs = point.point2D_idxs
    n = len(image_ids)
    if n < 2:
        continue

    random_idxs = list(range(n))   
    random.shuffle(random_idxs)
    patches = []
    for idx in random_idxs:
        image_id = image_ids[idx]
        point2D_idx = point2D_idxs[idx]
        image_name = images[image_id].name

        image = images_storage[image_name]

        (y, x) = images[image_id].xys[point2D_idx]
        if is_valid_patch(x, y, image):
            patch = get_patch(x, y, image)
            patches.append(patch)

        if len(patches) == 2:
            rgb[point_id] = (patches[0], patches[1])
            gray[point_id] = (patches[0][:, :, 0], patches[1][:, :, 0])
            break

logging.info(f'n_patches: {2 * len(rgb)}')
logging.info(f'used/total 3d points: {int(len(rgb))}/{len(pbar)}')
logging.info(f'unused 3d poitnts: {len(pbar) - int(len(rgb))}\n')
gray_file = os.path.join(OUTPUT, 'gray.npy')
logging.info(f'Saving gray data at: {gray_file}')
np.save(gray_file, gray)

###############################################################################
point_ids = list(gray.keys())
point_ids.sort()
info_file_path = os.path.join(OUTPUT, 'points.csv')
logging.info(f'Save info file at: {info_file_path}')
info = pd.DataFrame(point_ids, columns=['point_id'])
info.to_csv(info_file_path, index=None)
###############################################################################
"""
Generate test set
"""

BATCH_SIZE =  min(int(len(gray)), configs['batch_size'])
TEST_SIZE = configs['test_size']

logging.info(f'n_pos: {TEST_SIZE} - n_neg: {TEST_SIZE}')
logging.info(f'generated_batch_size: {BATCH_SIZE}\n')


test_set = generate_test_set(point_ids, TEST_SIZE, BATCH_SIZE, gray)
test_set_file = os.path.join(OUTPUT, f'{TEST_SIZE}_{TEST_SIZE}.npy')

logging.info(f'Save test set at {test_set_file}')
np.save(test_set_file, test_set)

logging.info(f'Completed\n')
