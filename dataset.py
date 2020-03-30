import os
import logging
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import random


class TripletPhotoTour(Dataset):
    def __init__(self, root: str, transform=None, test_scene='', train=True, train_scenes=[], n_triplets=-1): 
        self.root = root
        self.train = train
        self.transform = transform
        
        if self.train:
            self.train_scenes = train_scenes
            self.n_triplets = n_triplets
            self.train_scenes_info = merge_info(root=self.root, scenes=self.train_scenes)
            logging.info(f'num_3d_points: {self.train_scenes_info.shape[0]}')
            self.triplets = generate_triplets(self.train_scenes_info, self.n_triplets)
        else:
            self.test_scene = test_scene
            self.test_set = pd.read_csv(os.path.join(self.root, self.test_scene, '50000_50000.csv'))

    def __getitem__(self, index):
        if self.train:
            (scene_1, point_id1, patch_id1, scene_2, point_id2, patch_id2,
            scene_3, point_id3, patch_id3) = self.triplets.iloc[index]

            patch_1 = torch.tensor(get_patch(self.root, scene_1, point_id1, patch_id1))
            patch_2 = torch.tensor(get_patch(self.root, scene_2, point_id2, patch_id2))
            patch_3 = torch.tensor(get_patch(self.root, scene_3, point_id3, patch_id3))
            if self.transform is not None:
                patch_1 = self.transform(patch_1.numpy())
                patch_2 = self.transform(patch_2.numpy())
                patch_3 = self.transform(patch_3.numpy())
            return (patch_1, patch_2, patch_3)
        else:
            point_id1, patch_id1, point_id2, patch_id2, label = self.test_set.iloc[index]
            patch_1 = torch.tensor(get_patch(self.root, self.test_scene, point_id1, patch_id1))
            patch_2 = torch.tensor(get_patch(self.root, self.test_scene, point_id2, patch_id2))
            if self.transform is not None:
                patch_1 = self.transform(patch_1.numpy())
                patch_2 = self.transform(patch_2.numpy())
            return (patch_1, patch_2, torch.tensor(label))


    def __len__(self):
        if self.train:
            return self.triplets.shape[0]
        else:
            return self.test_set.shape[0]


def merge_info(root: str, scenes: list) -> pd.DataFrame:
    list_scenes = list(os.listdir(root))
    info_files = []
    for scene in scenes:
        info_file = pd.read_csv(os.path.join(root, scene, 'info.csv'))
        info_file['scene'] = scene
        info_files.append(info_file)
    return pd.concat(info_files)

def generate_triplets(train_scenes_info: pd.DataFrame, num_triplets: int) -> pd.DataFrame:
    triplets = []
    n_3d_points = train_scenes_info.shape[0]
    logging.info(f'generate {num_triplets} triplets')
    for x in tqdm(range(num_triplets)):
        idx1 = np.random.randint(0, n_3d_points)
        idx2 = np.random.randint(0, n_3d_points)
        while idx1 == idx2:
            idx2 = np.random.randint(0, n_3d_points)

        c = np.random.randint(0, 2)
        tup = (train_scenes_info['scene'].iloc[idx1], train_scenes_info['3d_point_id'].iloc[idx1], 0,
               train_scenes_info['scene'].iloc[idx1], train_scenes_info['3d_point_id'].iloc[idx1], 1,
               train_scenes_info['scene'].iloc[idx2], train_scenes_info['3d_point_id'].iloc[idx2], c)
        triplets.append(tup)
    columns = ['scene_1', '3d_point_id1', 'patch_id1',
               'scene_2', '3d_point_id2', 'patch_id2',
               'scene_3', '3d_point_id3', 'patch_id3']
    return pd.DataFrame(triplets, columns=columns)


def get_patch(root: str, scene: str, point_id: int, patch_id: int, gray=True) -> np.array:
    if gray:
        patch = cv2.imread(os.path.join(root, scene, 'gray', f'{point_id}.bmp'))
    else:
        patch = cv2.imread(os.path.join(root, scene, 'rgb', f'{point_id}.bmp'))
    return patch[:, 64*patch_id:64*patch_id+64, :]


class BrownTest(Dataset):
    def __init__(self, root: str, scene: str, transform=None): 
        self.data, self.labels, self.matches = torch.load(os.path.join(root, f'{scene}.pt'))
        self.transform = transform
    
    def __len__(self):
        return self.matches.size(0)
    
    
    def __getitem__(self, index):
        m = self.matches[index]
        img1 = self.data[m[0]].view(64, 64, 1)
        img1 = torch.cat((img1, img1, img1), axis=2)        
        img2 = self.data[m[1]].view(64, 64, 1)
        img2 = torch.cat((img2, img2, img2), axis=2)
        
        if self.transform is not None:
            img1 = self.transform(img1.numpy())
            img2 = self.transform(img2.numpy())
        return img1, img2, m[2]
              