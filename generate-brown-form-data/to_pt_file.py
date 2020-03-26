import logging
import deepdish as dd
import h5py
import os
from time import time
from tqdm import tqdm

from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
import torch

from colmap.scripts.python.read_write_model import read_model, qvec2rotmat
from colmap.scripts.python.read_dense import read_array

# logging.basicConfig(filename='logs.txt',
#                     filemode='a',
#                     format='%(asctime)s, %(levelname)s: %(message)s',
#                     datefmt='%y-%m-%d %H:%M:%S',
#                     level=logging.DEBUG)
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# logging.getLogger().addHandler(console)
logging.basicConfig(level=logging.INFO)

root = '.'
seq = 'buckingham_palace'
src = root + '/' + seq

cameras, images, points = read_model(path=src + '/dense/sparse', ext='.bin')


def is_valid_patch(x: float, y: float, patch_size, image: np.array):
    height = image.shape[0]
    width = image.shape[1]

    half_size = int(patch_size/2)
    if x - half_size < 0 or y - half_size < 0:
        return False
    if x - half_size + patch_size > height or y - half_size + patch_size > width:
        return False
    return True


def get_patch(x: float, y: float, patch_size: int, image: np.array):
    half_size = int(patch_size/2)
    
    x_start = int(x - half_size)
    y_start = int(y - half_size)
    return image[x_start: x_start+patch_size, y_start: y_start+patch_size, :]


PATCH_SIZE = 64
dataset = []

pbar = list(points.keys())
pbar.sort()
counter = 0
start = None
out_of_range = None
correct_patch = None
for point_id in tqdm(pbar):
    if counter == 0:
        out_of_range = 0
        correct_patch = 0
        start = point_id
    counter += 1
    point = points[point_id]
    image_ids = point.image_ids
    point2D_idxs = point.point2D_idxs
    n = len(image_ids)
    for idx in range(n):
        image_id = image_ids[idx]
        point2D_idx = point2D_idxs[idx]
        image_name = images[image_id].name
        try:
            # logging.info(point_id, idx)
            image = plt.imread(os.path.join(root, seq, 'dense/images', image_name))
            # logging.info(image.shape)
            (y, x) = images[image_id].xys[point2D_idx]
            if is_valid_patch(x, y, PATCH_SIZE, image):
                patch = get_patch(x, y, PATCH_SIZE, image)
                dataset.append({
                    'point_id': point_id,
                    'patch': patch
                })
                correct_patch += 1
            else:
                out_of_range += 1
        except FileNotFoundError as e:
            logging.info(e)
    if counter%8 == 0:
        filename = f'buckingham_palace_out/{start}_{point_id}.pt'
        with open(filename, 'wb') as f:
            torch.save(dataset, f)
        dataset.clear()
        counter = 0
        logging.info(f'Out of range patches / correct patches: {out_of_range}/{correct_patch}')
