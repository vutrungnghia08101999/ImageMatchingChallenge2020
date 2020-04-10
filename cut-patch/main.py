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

from utils import read_h5, is_valid_patch, get_patch

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

parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str)
args = parser.parse_args()

INPUT = '/media/vutrungnghia/New Volume/P2-ImageMatchingChallenge/baselines/imw-2020'
BASELINES_KPS = '/media/vutrungnghia/New Volume/P2-ImageMatchingChallenge/baselines/benchmark-patches-8k'
SCENE = args.scene
OUTPUT = '/media/vutrungnghia/New Volume/P2-ImageMatchingChallenge/dataset/evaluation/2048-hardnet'

logging.info(INPUT)
logging.info(BASELINES_KPS)
logging.info(SCENE)
logging.info(OUTPUT)

os.makedirs(os.path.join(OUTPUT, SCENE), exist_ok=False)
images_storage = {}
image_files = list(os.listdir(os.path.join(INPUT, SCENE)))
logging.info(f'Load {len(image_files)} images')
for filename in tqdm(image_files):
    images_storage[filename.split('.')[0]] = plt.imread(os.path.join(INPUT, SCENE, filename))

keypoints = read_h5(os.path.join(BASELINES_KPS, SCENE, 'keypoints.h5'))
patches = {}

for key in tqdm(keypoints.keys()):
    img = images_storage[key]
    kps = keypoints[key]

    idxs = []
    for idx in range(kps.shape[0]):
        x, y = kps[idx]
        if is_valid_patch(x, y, img):
            idxs.append(idx)
    random.shuffle(idxs)
    idxs = idxs[0:2048]
    keypoints[key] = keypoints[key][idxs]
    img_patches = []
    for idx in idxs:
        x, y = kps[idx]
        patch = get_patch(x, y, img)
        patch = patch[:, :, 0].reshape(1, 64, 64)
        img_patches.append(patch)
    patches[key] = np.concatenate(img_patches, axis=0)

h = h5py.File(os.path.join(OUTPUT, SCENE, 'keypoints.h5'), 'w')
for key in keypoints.keys():
    h.create_dataset(key, data=keypoints[key])
h.close()

h = h5py.File(os.path.join(OUTPUT, SCENE, 'patches.h5'), 'w')
for key in patches.keys():
    h.create_dataset(key, data=patches[key])
h.close()

logging.info('Completed\n')
