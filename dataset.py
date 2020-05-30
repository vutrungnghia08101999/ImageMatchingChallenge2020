import logging
import os
from tqdm import tqdm

import numpy as np
from random import shuffle
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

def load_data(root: str, scenes: list, n=2200) -> list:
    dataset = []
    logging.info('    Load training data')
    for scene in scenes:
        s = np.load(os.path.join(root, f'{scene}.npy'), allow_pickle=True).item()
        subset = []
        for k, v in s.items():
            subset.append(v)
        shuffle(subset)
        subset = subset[:int(n/len(scenes))]
        dataset = dataset + subset
        logging.info(f'      {scene}: {len(subset)}/{len(s)}')
    return dataset


# def transform_and_filter(pair: dict, top_k=2048) -> dict:
#     """get groundtruth matrix for a pair of images
#     Arguments:
#         pair {dict}:
#             matches: [(idx1, idx2), .... ]
#             keypoints: (N x 2, M x 2)
#             descriptors: (N x 128, M x 128)
#             scores: (N, M)
#             3dpoints: (N, M)
#             shape: ({width: , height: }, {width: , height: })
#             name: (str, str)
#     Returns:
#         np.array -- [description]
#     """
#     assert pair['keypoints'][0].shape[0] == pair['descriptors'][0].shape[0]
#     assert pair['keypoints'][0].shape[0] == pair['scores'][0].shape[0]
#     assert pair['keypoints'][0].shape[0] == pair['3dpoints'][0].shape[0]

#     assert pair['keypoints'][1].shape[0] == pair['descriptors'][1].shape[0]
#     assert pair['keypoints'][1].shape[0] == pair['scores'][1].shape[0]
#     assert pair['keypoints'][1].shape[0] == pair['3dpoints'][1].shape[0]

#     # get keypoints, descriptors and scores base on filter of has 3d points and descriptors
#     # has_3d_points0 = pair['3dpoints'][0] != -1
#     has_descriptors0 = ~np.isnan(pair['descriptors'][0][:, 0])
#     top_k_scores0_idxs = np.argsort(
#         -(pair['scores'][0] + has_descriptors0))[:top_k]
#     top_k_filter0 = np.array([False] * has_descriptors0.shape[0])
#     top_k_filter0[top_k_scores0_idxs] = True
#     filter0 = top_k_filter0 * has_descriptors0
#     keypoints0 = pair['keypoints'][0][filter0]
#     descriptors0 = pair['descriptors'][0][filter0]
#     scores0 = pair['scores'][0][filter0]

#     has_descriptors1 = ~np.isnan(pair['descriptors'][1][:, 0])
#     top_k_scores1_idxs = np.argsort(
#         -(pair['scores'][1] + has_descriptors1))[:top_k]
#     top_k_filter1 = np.array([False] * has_descriptors1.shape[0])
#     top_k_filter1[top_k_scores1_idxs] = True
#     filter1 = top_k_filter1 * has_descriptors1
#     keypoints1 = pair['keypoints'][1][filter1]
#     descriptors1 = pair['descriptors'][1][filter1]
#     scores1 = pair['scores'][1][filter1]

#     # construct the groundtruth matrix for keypoints after filtering
#     height, width = pair['keypoints'][0].shape[0] + 1, pair['keypoints'][1].shape[0] + 1
#     groundtruth = np.zeros((height, width))
#     for kp1_idx, kp2_idx in pair['matches']:
#         groundtruth[kp1_idx][kp2_idx] = 1

#     filter0 = np.append(filter0, True)
#     filter1 = np.append(filter1, True)
#     groundtruth = groundtruth[filter0, :]
#     groundtruth = groundtruth[:, filter1]

#     right_append = groundtruth.sum(axis=1) == 0
#     bottom_append = groundtruth.sum(axis=0) == 0
#     groundtruth[:, -1] += right_append
#     groundtruth[-1, :] += bottom_append
#     groundtruth[-1][-1] = 0

#     return {
#         'keypoints': (keypoints0, keypoints1),
#         'descriptors': (descriptors0.transpose(), descriptors1.transpose()),  # N x 128 => 128 x N, M x 128 => 128 x M
#         'scores': (scores0, scores1),
#         'shape': pair['shape'],
#         'name': pair['name'],
#         'groundtruth': groundtruth
#     }
def transform_and_filter(pair: dict, max_keypoints=2048) -> dict:
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
    if sum(filter0) > max_keypoints:
        true_idxs = np.argwhere(filter0 == True).squeeze()
        np.random.shuffle(true_idxs)
        reduced_true_idxs = true_idxs[:max_keypoints]
        filter0 = filter0 * False
        filter0[reduced_true_idxs] = True
    keypoints0 = pair['keypoints'][0][filter0]
    descriptors0 = pair['descriptors'][0][filter0]
    scores0 = pair['scores'][0][filter0]

    has_3d_points1 = pair['3dpoints'][1] != -1
    has_descriptors1 = ~np.isnan(pair['descriptors'][1][:, 0])
    filter1 = has_3d_points1 * has_descriptors1
    if sum(filter1) > max_keypoints:
        true_idxs = np.argwhere(filter1 == True).squeeze()
        np.random.shuffle(true_idxs)
        reduced_true_idxs = true_idxs[:max_keypoints]
        filter1 = filter1 * False
        filter1[reduced_true_idxs] = True
    keypoints1 = pair['keypoints'][1][filter1]
    descriptors1 = pair['descriptors'][1][filter1]
    scores1 = pair['scores'][1][filter1]

    # construct the groundtruth matrix for keypoints after filtering
    height, width = pair['keypoints'][0].shape[0] + 1, pair['keypoints'][1].shape[0] + 1
    groundtruth = np.zeros((height, width))
    for kp1_idx, kp2_idx in pair['matches']:
        groundtruth[kp1_idx][kp2_idx] = 1

    filter0 = np.append(filter0, True)
    filter1 = np.append(filter1, True)
    groundtruth = groundtruth[filter0, :]
    groundtruth = groundtruth[:, filter1]

    right_append = groundtruth.sum(axis=1) == 0
    bottom_append = groundtruth.sum(axis=0) == 0
    groundtruth[:, -1] += right_append
    groundtruth[-1, :] += bottom_append
    groundtruth[-1][-1] = 0

    return {
        'keypoints': (keypoints0, keypoints1),
        'descriptors': (descriptors0.transpose(), descriptors1.transpose()),  # N x 128 => 128 x N, M x 128 => 128 x M
        'scores': (scores0, scores1),
        'shape': pair['shape'],
        'name': pair['name'],
        'groundtruth': groundtruth
    }
