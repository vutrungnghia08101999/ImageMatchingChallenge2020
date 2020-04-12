import argparse
import logging
import os
from tqdm import tqdm

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from model import HardNet
from utils import cv2_scale, np_reshape, read_yaml, read_h5, to_3d_tensor, write_h5
from dataset import SubmissionDataset

logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
# logging.basicConfig(level=logging.INFO)

configs = read_yaml('configs.yml')
parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str)
args = parser.parse_args()
configs['scene'] = args.scene

logging.info(configs)

def create_transform():
    return transforms.Compose([transforms.Lambda(cv2_scale),
                               transforms.Lambda(np_reshape),
                               transforms.ToTensor(),
                               transforms.Normalize((configs['mean'],), (configs['std'],))])


model = HardNet()
checkpoint = torch.load(configs['checkpoint'], map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])

patches = read_h5(os.path.join(configs['patches'], configs['scene'], 'patches.h5'))

descriptors = {}
transform = create_transform()

logging.info(f'Go throught {len(patches)} images and embed n patches by Hardnet')
model.eval()
for key in tqdm(patches.keys()):
    dataset = SubmissionDataset(key, patches, create_transform())
    dataloader = DataLoader(dataset, 128, shuffle=False)
    bag = []
    with torch.no_grad():
        for batch in dataloader:
            descriptors_img = model(batch)
            bag.append(descriptors_img)
    bag = torch.cat(bag, axis=0)
    descriptors[key] = bag.numpy()

write_h5(os.path.join(configs['patches'], configs['scene'], 'descriptors.h5'), descriptors)

logging.info('Completed\n')
