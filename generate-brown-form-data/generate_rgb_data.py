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
INPUT = '/media/vutrungnghia/New Volume/P2-ImageMatchingChallenge/dataset/challenge-datas\
et/train/buckingham_palace/dense'
OUTPUT = 'rgb/buckingham_palace'
N_POS_NEG = 50000
#########################################


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


def concatenate_patches(patches: list):
    assert len(patches) <= 256
    cache = patches.copy()
    n = len(cache)
    for i in range(256 - n):
        cache.append(np.zeros((64, 64, 3), np.uint8) + 255)
    
    rows = []
    for row in range(16):
        rows.append(np.concatenate(cache[row * 16: row * 16 + 16], axis=1))
    return np.concatenate(rows, axis=0)


logging.info('=========================================================')
logging.info('\nGENERATE RGB PATCHES\n')
logging.info(f'input: {INPUT}')
logging.info(f'output: {OUTPUT}\n')

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

dataset = []
pbar = list(points.keys())
n_3d_points = len(pbar)
pbar.sort()
logging.info(f'Get 2 patches per 3d point of {len(pbar)} 3d points')
for index in tqdm(range(len(pbar))):
    point_id = pbar[index]
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
        if is_valid_patch(x, y, 64, image):
            patch = get_patch(x, y, 64, image)
            patches.append({
                'point_id': point_id,
                'patch': patch
            })

        if len(patches) == 2:
            dataset.append(patches[0])
            dataset.append(patches[1])
            break

logging.info(f'n_patches: {len(dataset)}')
logging.info(f'used/total 3d points: {int(len(dataset)/2)}/{len(pbar)}')
logging.info(f'unused 3d poitnts: {len(pbar) - int(len(dataset)/2)}\n')


##############################################################################
"""
generate info.txt, each row is 3d point's id
concatenate 256 patches to an image 1024 x 1024 and save.
"""

point_ids = []
patches = []
for tup in dataset:
    point_ids.append(tup['point_id'])
    patches.append(tup['patch'])

info_file = os.path.join(OUTPUT, 'info.txt')
if os.path.isfile(info_file):
    raise RuntimeError(f'{info_file} is already exist')
logging.info(f'Save info file at: {info_file}')
with open(info_file, 'w') as f:
    for item in point_ids:
        f.write(f"{item}\n")

n_batches = math.ceil(len(patches)/256)
logging.info(f'Concatenate and save {n_batches} images size 1024 x 1024')
for index in tqdm(range(n_batches)):
    image = concatenate_patches(patches[index * 256: index * 256 + 256])
    image = Image.fromarray(image)
    image_file = os.path.join(OUTPUT, f'{str(index).zfill(6)}.bmp')
    if os.path.isfile(image_file):
       raise RuntimeError(f'{image_file} is already exist') 
    image.save(image_file)

logging.info(f'unique_3d_ids: {len(pd.Series(point_ids).unique())}')
logging.info(f'n_patches: {len(point_ids)}')

###############################################################################
"""
Generate N_POS_NEG neg and N_POS_NEG pos pairs
Format:
patchID1   3DpointID1   unused1   patchID2   3DpointID2   unused2
"""

labels = point_ids
BATCH_SIZE = min(int(len(labels)/10), N_POS_NEG)

logging.info(f'n_pos: {N_POS_NEG} - n_neg: {N_POS_NEG}')
logging.info(f'generated_batch_size: {BATCH_SIZE}\n')


def generate_triplets(labels, num_triplets):
    def create_indices(_labels):
        inds = dict()
        for idx, ind in enumerate(_labels):
            if ind not in inds:
                inds[ind] = []
            inds[ind].append(idx)
        return inds

    triplets = []
    indices = create_indices(labels.numpy())
    unique_labels = np.unique(labels.numpy())
    n_classes = unique_labels.shape[0]
    # add only unique indices in batch
    already_idxs = set()

    for x in tqdm(range(num_triplets)):
        x = 0
        if len(already_idxs) >= BATCH_SIZE:
            already_idxs = set()
        c1 = unique_labels[np.random.randint(0, n_classes)]
        while c1 in already_idxs or len(indices[c1]) < 2:
            c1 = unique_labels[np.random.randint(0, n_classes)]
        already_idxs.add(c1)
        c2 = unique_labels[np.random.randint(0, n_classes)]
        while c1 == c2:
            c2 = unique_labels[np.random.randint(0, n_classes)]
        if len(indices[c1]) == 2:  # hack to speed up process
            n1, n2 = 0, 1
        else:
            n1 = np.random.randint(0, len(indices[c1]))
            n2 = np.random.randint(0, len(indices[c1]))
            while n1 == n2:
                n2 = np.random.randint(0, len(indices[c1]))
        n3 = np.random.randint(0, len(indices[c2]))
        # triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
        triplets.append([indices[c1][n1], c1, -1, indices[c1][n2], c1, -1])
        triplets.append([indices[c1][n1], c1, -1, indices[c2][n3], c2, -1])
    return torch.LongTensor(np.array(triplets))


test_set = generate_triplets(torch.tensor(labels), N_POS_NEG)
test_set_file = os.path.join(OUTPUT, f'{N_POS_NEG}_{N_POS_NEG}.txt')
if os.path.isfile(test_set_file):
    raise RuntimeError(f'{test_set_file} is already exist')
logging.info(f'Save test set file at: {test_set_file}')
np.savetxt(test_set_file, test_set.numpy(), fmt='%d')
logging.info('Conpleted\n')
