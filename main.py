#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import division, print_function

import argparse
import copy
from copy import deepcopy
import datetime
import logging
import math
import os
import random
import sys
import time
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eval_metrics import ErrorRateAt95Recall
from models import HardNet, ResNet, SOSNet32x32
from dataset import TripletPhotoTour, BrownTest
from losses import loss_HardNet, loss_L2Net
from utils import cv2_scale, np_reshape, read_yaml

logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
# logging.basicConfig(level=logging.INFO)


# In[ ]:


logging.info('\n\n================ IMAGE MATCHING CHALLENGE 2020 ==================\n\n')
configs = read_yaml('configs.yml')
os.makedirs(configs['model_dir'], exist_ok=True)


# In[ ]:


if configs['use_cuda']:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(configs['gpu_id'])
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


# In[ ]:


def train(train_loader, model, optimizer, epoch):
    model.train()
    for batch_id, data in tqdm(enumerate(train_loader)):
        if batch_id + 1 == len(train_loader):
            continue
        data_a, data_p = data

        if configs['use_cuda']:
            data_a, data_p  = data_a.cuda(), data_p.cuda()
        out_a = model(data_a)
        out_p = model(data_p)

        loss = loss_HardNet(out_a, out_p)
           
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_id%10 == 0:
            logging.info(f'{batch_id}/{len(train_loader)} - Loss: {loss.item()}')
        if batch_id%100 == 0:
            x = datetime.datetime.now()
            time = x.strftime("%y-%m-%d_%H:%M:%S")
            model_checkpoint = os.path.join(configs['model_dir'], f'checkpoint_{time}_{epoch}_{batch_id}.pth')
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()}, model_checkpoint)
            logging.info(model_checkpoint)
    logging.info(f'{len(train_loader)}/{len(train_loader)} - Loss: {loss.item()}')
    
    x = datetime.datetime.now()
    time = x.strftime("%y-%m-%d_%H:%M:%S")
    model_checkpoint = os.path.join(configs['model_dir'], f'checkpoint_{time}_{epoch}.pth')
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()}, model_checkpoint)
    logging.info(model_checkpoint)


def test(test_loader, model, epoch):
    model.eval()

    labels, distances = [], [] 
    for (data_a, data_p, label) in tqdm(test_loader):

        if configs['use_cuda']:
            data_a, data_p = data_a.cuda(), data_p.cuda()

        with torch.no_grad():
            out_a = model(data_a)
            out_p = model(data_p)
        dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy().reshape(-1,1))
        ll = label.data.cpu().numpy().reshape(-1, 1)
        labels.append(ll)

    num_tests = len(test_loader.dataset)
    labels = np.vstack(labels).reshape(num_tests)
    distances = np.vstack(distances).reshape(num_tests)

    fpr95 = ErrorRateAt95Recall(labels, 1.0 / (distances + 1e-8))
    logging.info('\33[91mTest set: Accuracy(FPR95): {:.8f}\n\33[0m'.format(fpr95))

    return


def create_transform():
    return transforms.Compose([#transforms.Lambda(cv2_scale),
                               #transforms.Lambda(np_reshape),
                               transforms.ToTensor(),
                               transforms.Normalize((configs['dataset']['mean'],), (configs['dataset']['std'],))])

def create_trainloader(root: str, train_scenes: list):   
    dataset = TripletPhotoTour(root=root,
                               transform=create_transform(),
                               train_scenes=train_scenes)
    return DataLoader(dataset, batch_size=configs['batch_size'], shuffle=True, num_workers=configs['n_workers'])

def create_testloader(root: str, test_scene: str, is_challenge_data: bool):
    if is_challenge_data:
        dataset = TripletPhotoTour(root=root,
                                   transform=create_transform(),
                                   test_scene=test_scene,
                                   train=False)
    else:
        dataset = BrownTest(root=root,
                            scene=test_scene,
                            transform=create_transform())
    return DataLoader(dataset, batch_size=configs['batch_size'], shuffle=False, num_workers=configs['n_workers'])


# In[ ]:


train_dataloader = create_trainloader(root=configs['dataset']['challenge_root'],
                                      train_scenes=configs['dataset']['train_scenes'])


# In[ ]:


test_dataloaders = []
for scene in configs['dataset']['test_scenes']['challenge']:
    logging.info(f'load test set: {scene}')
    dataloader = create_testloader(root=configs['dataset']['challenge_root'],
                                   test_scene=scene,
                                   is_challenge_data=True)
    test_dataloaders.append((scene, dataloader))


# In[ ]:


# model = ResNet()
model = HardNet()
# model = SOSNet32x32()
if configs['use_cuda']:
    model = model.cuda()

logging.info('\nTraining configurations: ')
logging.info(configs)


# In[ ]:


# optimizer = optim.SGD(model.features.parameters(), lr=configs['lr'],
#                       momentum=0.9, dampening=0.9,
#                       weight_decay=configs['weight_decay'])
optimizer = torch.optim.Adam(model.parameters(),
                             lr=configs['lr'],
                             betas=(0.9, 0.999),
                             eps=1e-08,
                             weight_decay=0,
                             amsgrad=False)

if configs['resume']:
    if os.path.isfile(configs['resume']):
        logging.info('=> loading checkpoint {}'.format(configs['resume']))
        checkpoint = torch.load(configs['resume'], map_location=torch.device('cpu'))
        configs['start_epoch'] = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
    else:
        logging.info('=> no checkpoint found at {}'.format(configs['resume']))


# In[ ]:


start = configs['start_epoch']
end = start + configs['epochs']
for epoch in range(start, end):
    logging.info(f'EPOCH: {epoch}')
    logging.info('========================')
    logging.info('Training sceces: ')
    logging.info(configs['dataset']['train_scenes'])
    train(train_dataloader, model, optimizer, epoch)
    
    for tup in test_dataloaders:
        logging.info(f'Test on {len(tup[1].dataset)} pairs of {tup[0]}')
        test(tup[1], model, epoch)
    
    train_dataloader = create_trainloader(root=configs['dataset']['challenge_root'],
                                          train_scenes=configs['dataset']['train_scenes'])
logging.info('Completed\n')


# In[ ]:




