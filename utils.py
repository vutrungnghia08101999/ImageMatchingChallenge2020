import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


def generate_test_set(points_ids: list, num_triplets:int, dataset: dict):
    n_3dpoints = len(points_ids)
    # assert num_triplets <= n_3dpoints
    point_inds1 = np.random.choice(n_3dpoints, num_triplets)
    point_inds2 = np.random.choice(n_3dpoints, num_triplets)
    
    for i, (id1, id2) in enumerate(tqdm(zip(point_inds1, point_inds2))):
        while id1 == id2:
            id2 = np.random.randint(0, n_3dpoints)
        point_inds2[i] = id2

    pos_test_set = [
        (dataset[points_ids[id1]][0], dataset[points_ids[id1]][1], 1)
    for id1 in point_inds1]
    
    neg_test_set = [
        (dataset[points_ids[id1]][1], dataset[points_ids[id2]][1], 0)
    for id1, id2 in zip(point_inds1, point_inds2)]

    return pos_test_set + neg_test_set


def generate_test_set_csv(points_ids: list, save_path):
    n_points = len(points_ids)
    points_ids1 = np.array(points_ids)

    pos_test_set = np.stack([
        points_ids1,
        np.zeros_like(points_ids1),
        points_ids1,
        np.ones_like(points_ids),
        np.ones_like(points_ids),
    ], axis=1)

    points_ids2 = points_ids1.copy()
    np.random.shuffle(points_ids2)
    for i, (id1, id2) in enumerate(tqdm(zip(points_ids1, points_ids2))):
        while id1 == id2:
            id2 = np.random.choice(points_ids1)
        points_ids2[i] = id2

    patch_id1 = np.random.choice(2, n_points)
    patch_id2 = np.random.choice(2, n_points)
    
    neg_test_set = np.stack([
        points_ids1,
        patch_id1,
        points_ids2,
        patch_id2,
        np.zeros_like(points_ids1),
    ], axis=1)
    
    test_set = np.concatenate([pos_test_set, neg_test_set], axis=0)
    df = pd.DataFrame(test_set, columns=[
        "point_id1", "patch_id1", "point_id2", "patch_id2", "label"
    ])

    df.to_csv(save_path, index=False)


def load_images(img_metadata, img_folder):
    """
    load all images of a scene
    
    args: 
        --img_meta_data: loaded from images.bin
        --img_folder: folder containing images to be loaded
    """
    img_dict = {}
    for img_id, datum in tqdm(img_metadata.items()):
        img_dict[img_id] = plt.imread(os.path.join(
            img_folder, 
            datum.name
        ))
    return img_dict

