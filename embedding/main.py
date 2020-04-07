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

patches = read_h5(os.path.join(configs['root'], configs['scene'], 'patches.h5'))

descriptors = {}
transform = create_transform()

logging.info('Go throught 100 images and embed 2000 patches by Hardnet')
model.eval()
for key in tqdm(patches.keys()):
    dataset = SubmissionDataset(key, patches, create_transform())
    dataloader = DataLoader(dataset, batch_size=len(dataset))
    with torch.no_grad():
        descriptors_img = model(iter(dataloader).next())
    descriptors[key] = descriptors_img.numpy()

write_h5(os.path.join(configs['root'], configs['scene'], 'descriptors.h5'), descriptors)

logging.info('Completed\n')