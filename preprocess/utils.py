from tqdm import tqdm
import yaml

import numpy as np
import pandas as pd


def read_yaml(file_path: str):
    with open(file_path, 'r') as stream:
        try:
            file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise RuntimeError(f'{file_path} does not exists!')
    return file


############################################################################
"""
Functions for generate step
"""


def is_valid_patch(x: float, y: float, image: np.array, patch_size=64) -> bool:
    height = image.shape[0]
    width = image.shape[1]

    half_size = int(patch_size/2)
    if x - half_size < 0 or y - half_size < 0:
        return False
    if x - half_size + patch_size > height or y - half_size + patch_size > width:
        return False
    return True


def get_patch(x: float, y: float, image: np.array, patch_size=64) -> np.array:
    half_size = int(patch_size/2)
    
    x_start = int(x - half_size)
    y_start = int(y - half_size)
    return image[x_start: x_start+patch_size, y_start: y_start+patch_size, :]


def concatenate_patches(patches: list) -> np.array:
    assert len(patches) <= 256
    cache = patches.copy()
    n = len(cache)
    for i in range(256 - n):
        cache.append(np.zeros((64, 64, 3), np.uint8) + 255)
    
    rows = []
    for row in range(16):
        rows.append(np.concatenate(cache[row * 16: row * 16 + 16], axis=1))
    return np.concatenate(rows, axis=0)


def generate_test_set(points_ids: list, num_triplets:int, batch_size: int):
    test_set = []
    n_3dpoints = len(points_ids)
    already_idxs = set()

    for x in tqdm(range(num_triplets)):
        x = 0
        if len(already_idxs) >= batch_size:
            already_idxs = set()
        point_id1 = points_ids[np.random.randint(0, n_3dpoints)]
        while point_id1 in already_idxs:
            point_id1 = points_ids[np.random.randint(0, n_3dpoints)]
        already_idxs.add(point_id1)
        point_id2 = points_ids[np.random.randint(0, n_3dpoints)]
        while point_id1 == point_id2:
            point_id2 = points_ids[np.random.randint(0, n_3dpoints)]

        pos_patch_id = np.random.randint(0, 2)
        neg_patch_id = np.random.randint(0, 2)

        test_set.append([point_id1, pos_patch_id, point_id2, neg_patch_id, 0])
        test_set.append([point_id1, 0, point_id1, 1, 1])
    # return triplets
    return pd.DataFrame(test_set, columns=['3d_point_id1', 'patch_id1', '3d_point_id2', 'patch_id2', 'label'])
