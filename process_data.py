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
from utils import is_valid_patch, get_patch, concatenate_patches, read_yaml, generate_test_set
from constants import (
    SCENE_DIR,
    SCENE_NAME,
    OUTPUT,
    N_POS_NEG_TEST,
    GENERATED_TEST_BATCH_SIZE,
)

logging.basicConfig(filename='logs_test.txt',
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
SCENE_OUTPUT_DIR = os.path.join(configs[OUTPUT], configs[SCENE_NAME])
RGB_DIR = os.path.join(SCENE_OUTPUT_DIR, 'rgb')
GRAY_DIR = os.path.join(SCENE_OUTPUT_DIR, 'gray')
#########################################
if not os.path.isdir(configs[SCENE_DIR]):
    raise RuntimeError(f'{os.path.isdir(configs[SCENE_DIR])} does not exist!')
if not os.path.isdir(configs[OUTPUT]):
    raise RuntimeError(f'{configs[OUTPUT]} does not exist!')
os.makedirs(SCENE_OUTPUT_DIR, exist_ok=False)
os.makedirs(RGB_DIR, exist_ok=False)
os.makedirs(GRAY_DIR, exist_ok=False)

###########################################################################
"""
Go throught list of 3d points and cut exactly 2 patches and store to dataset
"""

cameras, images, points = read_model(path=os.path.join(configs[SCENE_DIR], 'sparse'), ext='.bin')

images_storage = {}
image_files = list(os.listdir(os.path.join(configs[SCENE_DIR], 'images')))
logging.info(f'Load {len(image_files)} images')
for filename in tqdm(image_files):
    images_storage[filename] = plt.imread(os.path.join(configs[SCENE_DIR], 'images', filename))

dataset = []
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
            dataset.append({
                '3d_point_id': point_id,
                'patch_1': patches[0],
                'patch_2': patches[1]
            })
            break

logging.info(f'n_patches: {2 * len(dataset)}')
logging.info(f'used/total 3d points: {int(len(dataset))}/{len(pbar)}')
logging.info(f'unused 3d poitnts: {len(pbar) - int(len(dataset))}\n')


# return dataset
##############################################################################
"""
generate info.txt, each row is 3d point's id
concatenate 256 patches to an image 1024 x 1024 and save.
"""

point_ids = []
for tup in dataset:
    point_ids.append(tup['3d_point_id'])

info_file_path = os.path.join(SCENE_OUTPUT_DIR, 'info.csv')
if os.path.isfile(info_file_path):
    raise RuntimeError(f'{info_file_path} is already exist')
logging.info(f'Save info file at: {info_file_path}')
info = pd.DataFrame(point_ids, columns=['3d_point_id'])
info.to_csv(info_file_path, index=None)

for patches in tqdm(dataset):
    point_id = patches['3d_point_id']
    patch_1 = patches['patch_1']
    patch_2 = patches['patch_2']
    rgb_image = np.concatenate([patch_1, patch_2], axis=1)
    gray_image = rgb_image[:, :, 0]

    rgb_image = Image.fromarray(rgb_image)
    rgb_file = os.path.join(RGB_DIR, f'{point_id}.bmp')
    if os.path.isfile(rgb_file):
        raise RuntimeError(f'{rgb_file} is already exist')
    rgb_image.save(rgb_file)

    gray_image = Image.fromarray(gray_image)
    gray_file = os.path.join(GRAY_DIR, f'{point_id}.bmp')
    if os.path.isfile(gray_file):
        raise RuntimeError(f'{gray_file} is already exist')
    gray_image.save(gray_file)

# return point_ids
###############################################################################
"""
Generate test set
"""

batch_size =  min(int(len(point_ids)/2), configs[GENERATED_TEST_BATCH_SIZE])

logging.info(f'n_pos: {configs[N_POS_NEG_TEST]} - n_neg: {configs[N_POS_NEG_TEST]}')
logging.info(f'generated_batch_size: {batch_size}\n')




test_set = generate_test_set(point_ids, configs[N_POS_NEG_TEST], batch_size)
test_set_file = os.path.join(SCENE_OUTPUT_DIR, 
                             f'{configs[N_POS_NEG_TEST]}_{configs[N_POS_NEG_TEST]}.csv')

if os.path.isfile(test_set_file):
    raise RuntimeError(f'{test_set_file} is already exist')
logging.info(f'Save test set file at {test_set_file}')
test_set.to_csv(test_set_file, index=None)

logging.info(f'Completed\n')
