import argparse
import logging
import h5py
import os
from PIL import Image
import random
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import read_h5, is_valid_patch, get_patch, read_yaml

logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
# logging.basicConfig(level=logging.INFO)

np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str)
args = parser.parse_args()

configs = read_yaml('configs.yml')
configs['scene'] = args.scene
IMAGES = configs['images']
KPS = configs['kps']
SCENE = configs['scene']
OUTPUT = configs['output']

logging.info(configs)

os.makedirs(os.path.join(OUTPUT, SCENE), exist_ok=False)
images_storage = {}
image_files = list(os.listdir(os.path.join(IMAGES, SCENE)))
logging.info(f'Load {len(image_files)} images')
for filename in tqdm(image_files):
    images_storage[filename.split('.')[0]] = plt.imread(os.path.join(IMAGES, SCENE, filename))

keypoints = read_h5(os.path.join(KPS, SCENE, 'keypoints.h5'))
patches = {}

for key in tqdm(keypoints.keys()):
    img = images_storage[key]
    kps = keypoints[key]

    idxs = []
    for idx in range(kps.shape[0]):
        x, y = kps[idx]
        if is_valid_patch(x, y, img):
            idxs.append(idx)

    keypoints[key] = keypoints[key][idxs]
    img_patches = []
    for idx in range(keypoints[key].shape[0]):
        x, y = keypoints[key][idx]
        patch = get_patch(x, y, img)
        patch = patch[:, :, 0].reshape(1, 64, 64)
        img_patches.append(patch)
    patches[key] = np.concatenate(img_patches, axis=0)

n_kps = []
for key in keypoints.keys():
    if keypoints[key].shape[0] != patches[key].shape[0]:
        raise RuntimeError(f'{key} incorrect')
    n_kps.append(keypoints[key].shape[0])

logging.info(f'max n kps: {max(n_kps)}')
logging.info(f'min n kps: {min(n_kps)}')
logging.info(f'average n kps: {sum(n_kps)/len(n_kps)}')
# for key in keypoints.keys():
#     keypoints[key] = keypoints[key][0:n]
#     patches[key] = patches[key][0:n]

h = h5py.File(os.path.join(OUTPUT, SCENE, 'keypoints.h5'), 'w')
for key in keypoints.keys():
    h.create_dataset(key, data=keypoints[key])
h.close()

h = h5py.File(os.path.join(OUTPUT, SCENE, 'patches.h5'), 'w')
for key in patches.keys():
    h.create_dataset(key, data=patches[key])
h.close()

logging.info('Completed\n')
