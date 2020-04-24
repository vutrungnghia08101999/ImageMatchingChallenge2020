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
    def __init__(self, root: str, transform=None, test_scene='', train=True, train_scenes=[]): 
        self.transform = transform
        self.train = train
        
        if self.train:
            self.dataset = load_train_dataset(root, train_scenes)
            self.train_scenes_info = merge_info(root=root, scenes=train_scenes)
            logging.info(f'num_3d_points: {self.train_scenes_info.shape[0]}')
            self.triplets = generate_triplets(self.train_scenes_info)
        else:
            self.test_set = load_test_dataset(root=root, test_scene=test_scene)

    def __getitem__(self, index):
        if self.train:
            (scene_1, point_id1, patch_id1, scene_2, point_id2, patch_id2) = self.triplets.iloc[index]

            patch_1 = torch.tensor(self.dataset[scene_1][point_id1][patch_id1])
            patch_2 = torch.tensor(self.dataset[scene_2][point_id2][patch_id2])
            if self.transform is not None:
                patch_1 = self.transform(patch_1.numpy())
                patch_2 = self.transform(patch_2.numpy())
            return (patch_1, patch_2)
        else:
            patch_1, patch_2, label = self.test_set[index]
            patch_1 = torch.tensor(patch_1)
            patch_2 = torch.tensor(patch_2)
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
        info_file = pd.read_csv(os.path.join(root, scene, 'points.csv'))
        info_file['scene'] = scene
        info_files.append(info_file)
    return pd.concat(info_files)

def generate_triplets(train_scenes_info: pd.DataFrame) -> pd.DataFrame:
    triplets = []
    n_3d_points = train_scenes_info.shape[0]
    logging.info(f'generate {n_3d_points} triplets (simple add a negative patch to a matched patch)')
    for idx1 in tqdm(range(n_3d_points)):
        tup = (train_scenes_info['scene'].iloc[idx1], train_scenes_info['point_id'].iloc[idx1], 0,
               train_scenes_info['scene'].iloc[idx1], train_scenes_info['point_id'].iloc[idx1], 1)
        triplets.append(tup)
    columns = ['scene_1', 'point_id1', 'patch_id1',
               'scene_2', 'point_id2', 'patch_id2']
    df = pd.DataFrame(triplets, columns=columns)
    logging.info('10 first triplets: ')
    logging.info(df.head(10))
    logging.info('10 last triplets: ')
    logging.info(df.tail(10))
    return df

def load_train_dataset(root: str, scenes: list):
    dataset = {}
    for scene in scenes:
        dic = np.load(os.path.join(root, scene, 'rgb.npy'), allow_pickle=True)
        dic = dic.item()
        logging.info(f'Load {scene} dataset with n_points: {len(dic)}')
        dataset[scene] = dic
    return dataset

def load_test_dataset(root: str, test_scene: str):
    return np.load(os.path.join(root, test_scene, '50000_50000_rgb.npy'), allow_pickle=True)

def to_3d_tensor(s: torch.tensor):
    s = s.view(64, 64, 1)
    return torch.cat((s, s, s), axis=2)

class BrownTest(Dataset):
    def __init__(self, root: str, scene: str, transform=None): 
        self.data, self.labels, self.matches = torch.load(os.path.join(root, f'{scene}.pt'))
        self.transform = transform
    
    def __len__(self):
        return self.matches.size(0)
    
    
    def __getitem__(self, index):
        m = self.matches[index]
        img1 = to_3d_tensor(self.data[m[0]])        
        img2 = to_3d_tensor(self.data[m[1]])
        
        if self.transform is not None:
            img1 = self.transform(img1.numpy())
            img2 = self.transform(img2.numpy())
        return img1, img2, m[2]
