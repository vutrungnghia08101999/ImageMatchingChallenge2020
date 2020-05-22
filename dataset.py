import os
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn


# root = '/media/vutrungnghia/New Volume/P2-ImageMatchingChallenge/dataset/superglue/output'
# scenes = ['reichstag', 'brandenburg_gate']


class SuperGlueDataset(Dataset):
    def __init__(self, root: str, scenes: str):
        self.pairs = load_data(root, scenes)
    
    def __len__(self):
        return len(self.pairs)
    
    
    def __getitem__(self, index):
        pair = self.pairs[index]
        return transform_and_filter(pair)

def load_data(root: str, scenes: list) -> list:
    dataset = []
    for scene in tqdm(scenes):
        s = np.load(os.path.join(root, f'{scene}.npy'), allow_pickle=True).item()
        for k, v in s.items():
            dataset.append(v)
    return dataset


def transform_and_filter(pair: dict) -> dict:
    """get groundtruth matrix for a pair of images
    Arguments:
        pair {dict}:
            matches: [(idx1, idx2), .... ]
            keypoints: (N x 2, M x 2)
            descriptors: (N x 128, M x 128)
            scores: (N, M)
            3dpoints: (N, M)
            shape: ({width: , height: }, {width: , height: })
            name: (str, str)
    Returns:
        np.array -- [description]
    """
    assert pair['keypoints'][0].shape[0] == pair['descriptors'][0].shape[0]
    assert pair['keypoints'][0].shape[0] == pair['scores'][0].shape[0]
    assert pair['keypoints'][0].shape[0] == pair['3dpoints'][0].shape[0]

    assert pair['keypoints'][1].shape[0] == pair['descriptors'][1].shape[0]
    assert pair['keypoints'][1].shape[0] == pair['scores'][1].shape[0]
    assert pair['keypoints'][1].shape[0] == pair['3dpoints'][1].shape[0]

    # get keypoints, descriptors and scores base on filter of has 3d points and descriptors
    has_3d_points0 = pair['3dpoints'][0] != -1
    has_descriptors0 = ~np.isnan(pair['descriptors'][0][:, 0])
    filter0 = has_3d_points0 * has_descriptors0
    keypoints0 = pair['keypoints'][0][filter0]
    descriptors0 = pair['descriptors'][0][filter0]
    scores0 = pair['scores'][0][filter0]

    has_3d_points1 = pair['3dpoints'][1] != -1
    has_descriptors1 = ~np.isnan(pair['descriptors'][1][:, 0])
    filter1 = has_3d_points1 * has_descriptors1
    keypoints1 = pair['keypoints'][1][filter1]
    descriptors1 = pair['descriptors'][1][filter1]
    scores1 = pair['scores'][1][filter1]

    # construct the groundtruth matrix for keypoints after filtering
    height, width = pair['keypoints'][0].shape[0] + 1, pair['keypoints'][1].shape[0] + 1
    groundtruth = np.zeros((height, width))
    right_append = np.ones(height)
    bottom_append = np.ones(width)

    for kp1_idx, kp2_idx in pair['matches']:
        groundtruth[kp1_idx][kp2_idx] = 1
        right_append[kp1_idx] = 0
        bottom_append[kp2_idx] = 0

    groundtruth[:, width - 1] += right_append
    groundtruth[height - 1, :] += bottom_append
    groundtruth[height - 1][width - 1] = 0

    filter0 = np.append(filter0, True)
    filter1 = np.append(filter1, True)
    groundtruth = groundtruth[filter0, :]
    groundtruth = groundtruth[:, filter1]

    return {
        'keypoints': (keypoints0, keypoints1),
        'descriptors': (descriptors0, descriptors1),
        'scores': (scores0, scores1),
        'shape': pair['shape'],
        'name': pair['name'],
        'groundtruth': groundtruth
    }
