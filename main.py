
import argparse
import datetime
import itertools
import logging
import os
import random
from tqdm import tqdm
import cv2

import numpy as np
import torch
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader

from model import SuperGlue
from utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, AverageMeter, write_h5, AverageMeter)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
torch.manual_seed(0)

logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
# logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--sinkhorn_iterations', type=int, default=20,
                    help='Number of Sinkhorn iterations performed by SuperGlue')
parser.add_argument('--match_threshold', type=float, default=0.2,
                    help='SuperGlue match threshold')
parser.add_argument('--weights', type=str, default='/media/vutrungnghia/New Volume/P2-ImageMatchingChallenge/models/checkpoint_20-05-29_01-45-07_0.pth')
parser.add_argument('--valid_data', type=str,
                    default='/media/vutrungnghia/New Volume/P2-ImageMatchingChallenge/dataset/superglue/challenge-valid/val_2048.npy')
parser.add_argument('--output', type=str, default='/media/vutrungnghia/New Volume/P2-ImageMatchingChallenge/dataset/superglue/challenge-valid/output')
args = parser.parse_args()
logging.info(f'\n========= IMAGE MATCHING CHALLENGE 2020 ==========\n')
logging.info(args._get_kwargs())

model = SuperGlue({'sinkhorn_iterations': args.sinkhorn_iterations,
                   'match_threshold': args.match_threshold})
if torch.cuda.is_available():
    checkpoint = torch.load(args.weights)
    model = model.cuda()
else:
    checkpoint = torch.load(args.weights, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
logging.info(f'Load model at: {args.weights}')

valid_data = np.load(args.valid_data, allow_pickle=True).item()

model.eval()
for scene in valid_data.keys():
    logging.info(f'SCENE: {scene}')
    scene_folder = os.path.join(args.output, scene)
    os.makedirs(scene_folder, exist_ok=True)
    local_features = valid_data[scene]
    avg_matches = AverageMeter()
    # ######################### dev #############################
    # tmp = {}
    # keys = list(local_features)[:4]
    # for key in keys:
    #     tmp[key] = local_features[key]
    # local_features = tmp
    # ###########################################################
    pairs = itertools.combinations(sorted(list(local_features.keys()), reverse=True), 2)
    pairs = list(pairs)
    matches_keypoints = {}
    for img0, img1 in tqdm(pairs):
        key = img0.split('.')[0] + '-' + img1.split('.')[0]
        features0 = valid_data[scene][img0]
        features1 = valid_data[scene][img1]
        inputs = {
            'shape0': {
                'height': torch.tensor(features0['shape']['height']),
                'width': torch.tensor(features0['shape']['width'])},
            'descriptors0': torch.from_numpy(features0['descriptors'].transpose()).unsqueeze(0).cuda(),
            'keypoints0': torch.from_numpy(features0['keypoints']).unsqueeze(0).cuda(),
            'scores0': torch.from_numpy(features0['scores']).unsqueeze(0).cuda(),
            'shape1': {
                'height': torch.tensor(features1['shape']['height']),
                'width': torch.tensor(features1['shape']['width'])},
            'descriptors1': torch.from_numpy(features1['descriptors'].transpose()).unsqueeze(0).cuda(),
            'keypoints1': torch.from_numpy(features1['keypoints']).unsqueeze(0).cuda(),
            'scores1': torch.from_numpy(features1['scores']).unsqueeze(0).cuda()
        }

        with torch.no_grad():
            pred = model(inputs, is_train=False)
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        matches = pred['matches0']

        indices_img0 = np.argwhere(matches != -1)
        avg_matches.update(indices_img0.shape[0])
        indices_img1 = matches[indices_img0]
        links = np.stack([indices_img0, indices_img1], axis=1)
        links = links[:, :, 0].transpose()
        matches_keypoints[key] = links
    logging.info(f'avg_matches: {avg_matches.avg}')
    
    keypoints = {}
    descriptors = {}

    for image_name in local_features.keys():
        key = image_name.split('.')[0]
        keypoints[key] = local_features[image_name]['keypoints']
        descriptors[key] = local_features[image_name]['descriptors']
    
    write_h5(os.path.join(scene_folder, 'keypoints.h5'), keypoints)
    write_h5(os.path.join(scene_folder, 'descriptors.h5'), descriptors)
    write_h5(os.path.join(scene_folder, 'matches_keypoints.h5'), matches_keypoints)
