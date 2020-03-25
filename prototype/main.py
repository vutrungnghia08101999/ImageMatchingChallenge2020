#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division, print_function

import argparse
import copy
from copy import deepcopy
import datetime
import logging
import math
import os
import PIL
from PIL import Image
import random
import sys
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
from models import HardNet
from dataset import TripletPhotoTour
from losses import loss_HardNet
from utils import cv2_scale, np_reshape, read_yaml
from constants import (
    AUGMENTATION,
    ANCHORAVE,
    ANCHORSWAP,
    BATCH_SIZE,
    BATCH_REDUCE,
    DATASET,
    DATAROOT,
    EPOCHS,
    ENVIRONMENT,
    EXPERIMENT_NAME,
    FLIPROT,
    GPU_ID,
    LR_DECAY,
    LOSS,
    LEARNING_RATE,
    LOG_INTERVAL,
    MODEL_DIR,
    MARGIN,
    N_TRIPLETS,
    NUM_WORKERS,
    RESUME,
    SEED,
    START_EPOCH,
    SET_1,
    SET_2,
    TRAIN_MEAN_IMAGE,
    TRAIN_STD_IMAGE,
    TRAINING_SET,
    TEST_SET,
    TEST_BATCH_SIZE,
    USE_CUDA,
    WEIGHT_DECAY,
)

logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
# logging.basicConfig(level=logging.INFO)


# In[ ]:


configs = read_yaml('configs.yml')


# In[ ]:


models_output = os.path.join('models', f'{configs[EXPERIMENT_NAME]}')

os.environ['CUDA_VISIBLE_DEVICES'] = str(configs[ENVIRONMENT][GPU_ID])

if configs[ENVIRONMENT][USE_CUDA]:
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(configs[ENVIRONMENT][SEED])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(configs[ENVIRONMENT][SEED])
torch.manual_seed(configs[ENVIRONMENT][SEED])
np.random.seed(configs[ENVIRONMENT][SEED])


# In[ ]:


def train(train_loader, model, optimizer, epoch):
    # switch to train mode
    model.train()
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, data in pbar:

        data_a, data_p = data

        data_a, data_p  = data_a.cuda(), data_p.cuda()
        if configs[ENVIRONMENT][USE_CUDA]:
            data_a, data_p  = data_a.cuda(), data_p.cuda()
            out_a = model(data_a)
            out_p = model(data_p)
        else:
            out_a = model(data_a)
            out_p = model(data_p)           

        loss = loss_HardNet(out_a, out_p,
                        margin=configs[MARGIN],
                        anchor_swap=configs[ANCHORSWAP],
                        anchor_ave=configs[ANCHORAVE],
                        batch_reduce = configs[BATCH_REDUCE],
                        loss_type = configs[LOSS])
           
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adjust_learning_rate(optimizer)
        if batch_idx % configs[ENVIRONMENT][LOG_INTERVAL] == 0:
            pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                 epoch, batch_idx * len(data_a), len(train_loader.dataset),
                                 100. * batch_idx / len(train_loader),
                                 loss.item()))

    try:
        os.stat(f'{models_output}')
    except:
        os.makedirs(f'{models_output}')
    
    x = datetime.datetime.now()
    time = x.strftime("%y-%m-%d_%H:%M:%S")
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()}, f'{models_output}/checkpoint_{epoch}_{time}.pth')
    logging.info(f'{models_output}/checkpoint_{time}_{epoch}.pth')


def test(test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:

        if configs[ENVIRONMENT][USE_CUDA]:
            data_a, data_p = data_a.cuda(), data_p.cuda()

        with torch.no_grad():
            out_a = model(data_a)
            out_p = model(data_p)
        dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy().reshape(-1,1))
        ll = label.data.cpu().numpy().reshape(-1, 1)
        labels.append(ll)

        if batch_idx % configs[ENVIRONMENT][LOG_INTERVAL] == 0:
            pbar.set_description(' Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset),
                       100. * batch_idx / len(test_loader)))

    num_tests = test_loader.dataset.matches.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    distances = np.vstack(distances).reshape(num_tests)

    fpr95 = ErrorRateAt95Recall(labels, 1.0 / (distances + 1e-8))
    logging.info('\33[91mTest set: Accuracy(FPR95): {:.8f}\n\33[0m'.format(fpr95))

    return

def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        group['lr'] = configs[LEARNING_RATE] * (
        1.0 - float(group['step']) * float(configs[BATCH_SIZE]) / (configs[N_TRIPLETS] * float(configs[EPOCHS])))
    return

def create_dataloader(name: str, is_train: bool, load_random_triplet: bool):
    transform = transforms.Compose([
        transforms.Lambda(cv2_scale),
        transforms.Lambda(np_reshape),
        transforms.ToTensor(),
        transforms.Normalize((configs[DATASET][TRAIN_MEAN_IMAGE],), (configs[DATASET][TRAIN_STD_IMAGE],))])
    
    dataset = TripletPhotoTour(n_triplets=configs[N_TRIPLETS],
                               fliprot = configs[FLIPROT],
                               train=is_train,
                               load_random_triplets = load_random_triplet,
                               batch_size=configs[BATCH_SIZE],
                               root=configs[DATASET][DATAROOT],
                               name=name,
                               transform=transform)
    return DataLoader(dataset, batch_size=configs[BATCH_SIZE], shuffle=False, num_workers=configs[NUM_WORKERS])


# In[ ]:


reichstag_dataloader = create_dataloader(name='reichstag',
                                       is_train=True,
                                       load_random_triplet=False)
notredame_dataloader = create_dataloader(name='notredame',
                                       is_train=False,
                                       load_random_triplet=False)
yosemite_dataloader = create_dataloader(name='yosemite',
                                       is_train=False,
                                       load_random_triplet=False)
# train_dataloader = create_dataloader(name='reichstag',
#                                        is_train=True,
#                                        load_random_triplet=False)
# test_dataloader = create_dataloader(name='reichstag',
#                                        is_train=False,
#                                        load_random_triplet=False)


# In[ ]:


model = HardNet()
if configs[ENVIRONMENT][USE_CUDA]:
    model = model.cuda()
logging.info(configs)


# In[ ]:


optimizer = optimizer = optim.SGD(model.features.parameters(), lr=configs[LEARNING_RATE],
                                  momentum=0.9, dampening=0.9,
                                  weight_decay=configs[WEIGHT_DECAY])
# optionally resume from a checkpoint
if configs[RESUME]:
    if os.path.isfile(configs[RESUME]):
        logging.info('=> loading checkpoint {}'.format(configs[RESUME]))
        checkpoint = torch.load(configs[RESUME])
        configs[START_EPOCH] = checkpoint['epoch']
        checkpoint = torch.load(configs[RESUME])
        model.load_state_dict(checkpoint['state_dict'])
    else:
        logging.info('=> no checkpoint found at {}'.format(configs[RESUME]))


# In[ ]:


start = configs[START_EPOCH]
end = start + configs[EPOCHS]
for epoch in range(start, end):
    train(reichstag_dataloader, model, optimizer, epoch)
    logging.info('Test on notredame dataset')
    test(notredame_dataloader, model, epoch)
    logging.info('Test on yosemite dataset')
    test(yosemite_dataloader, model, epoch)
    
    reichstag_dataloader = create_dataloader(name='reichstag',
                                           is_train=True,
                                           load_random_triplet=False)


# In[ ]:




