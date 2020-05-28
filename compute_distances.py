import h5py
import os
from tqdm import tqdm

import numpy as np

ROOT = '/media/vutrungnghia/New Volume/P2-ImageMatchingChallenge/dataset/superglue/challenge-valid/submission/sacre_coeur'

filename = "file.hdf5"

def read_h5(path: str):
    data = {}
    with h5py.File(path, "r") as f:
        for k in tqdm(f.keys()):
            data[k] = f[k].value
    return data

keypoints = read_h5(os.path.join(ROOT, 'keypoints.h5'))
matches_keypoints = read_h5(os.path.join(ROOT, 'matches_keypoints.h5'))
desciptors = read_h5(os.path.join(ROOT, 'descriptors.h5'))

summary = []

for pair in tqdm(matches_keypoints.keys()):
    img0 = pair.split('-')[0]
    img1 = pair.split('-')[1]

    desciptors0 = desciptors[img0]
    desciptors1 = desciptors[img1]

    for idx0, idx1 in matches_keypoints[pair].transpose():
        u = desciptors0[idx0]
        v = desciptors1[idx1]
        dis = np.sqrt(np.sum((u - v)**2))
        summary.append(dis)

print(f'avg distance: {sum(summary)/len(summary)}')
