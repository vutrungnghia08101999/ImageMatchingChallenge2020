import argparse
import logging
import random
from tqdm import tqdm

import matplotlib.cm as cm
import numpy as np
import torch
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader

from dataset import SuperGlueDataset
from model import SuperGlue
from utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, AverageMeter)

logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
# logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str,
                    default='/media/vutrungnghia/New Volume/P2-ImageMatchingChallenge/dataset/superglue/train/output')
parser.add_argument('--train_scenes', type=str,
                    default='reichstag')
parser.add_argument('--sinkhorn_iterations', type=int, default=20,
                    help='Number of Sinkhorn iterations performed by SuperGlue')
parser.add_argument('--match_threshold', type=float, default=0.2,
                    help='SuperGlue match threshold')
parser.add_argument('--weights', type=str,
                    default='')
parser.add_argument('--valid_data', type=str,
                    default='/media/vutrungnghia/New Volume/P2-ImageMatchingChallenge/dataset/superglue/valid/val_2048.npy')
parser.add_argument('--valid_gt', type=str,
                    default='valid/yfcc_sample_pairs_with_gt.txt')
parser.add_argument('--n_epochs', type=int, default=50)
args = parser.parse_args()
logging.info(f'\n========= IMAGE MATCHING CHALLENGE 2020 ==========\n')
logging.info(args._get_kwargs())

# creat train dataset and dataloader, load valid data and valid gt
dataset = SuperGlueDataset(args.root, args.train_scenes.split(','))
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
valid_data = np.load(args.valid_data, allow_pickle=True).item()
with open(args.valid_gt, 'r') as f:
    valid_gt = [l.split() for l in f.readlines()]
logging.info(f'VALID_SCENES: {valid_data.keys()}')

# create superglue model and load checkpoint if exist
model = SuperGlue({'sinkhorn_iterations': args.sinkhorn_iterations,
                   'match_threshold': args.match_threshold})
if torch.cuda.is_available():
    model = model.cuda()
if args.weights:
    checkpoint = torch.load(args.weights)
    model.load_state_dict(checkpoint)

# create optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.001,
                             betas=(0.9, 0.999))

for epoch in range(args.n_epochs):
    logging.info(f'EPOCH: {epoch+1}/{args.n_epochs}')
    # ******************* TRAINING PHASE ***********************
    logging.info(f'TRAINING PHASE:')
    model.train()
    avg_loss = AverageMeter()
    for pair in dataloader:
        groundtruth = pair['groundtruth']
        inputs = {
            'shape0': pair['shape'][0],
            'descriptors0': pair['descriptors'][0],
            'keypoints0': pair['keypoints'][0],
            'scores0': pair['scores'][0],
            'shape1': pair['shape'][1],
            'descriptors1': pair['descriptors'][1],
            'keypoints1': pair['keypoints'][1],
            'scores1': pair['scores'][1]
        }

        log_matrix = model(inputs)  # log_matrix.exp() satifys: sum(row) = sum(col) = 1
        if torch.cuda.is_available():
            loss = -1.0 * (log_matrix * groundtruth.type(torch.cuda.FloatTensor)).sum()
        else:
            loss = -1.0 * (log_matrix * groundtruth.type(torch.FloatTensor)).sum()
        avg_loss.update(loss.item()/groundtruth.sum().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logging.info(f'avg loss: {avg_loss.avg}')

    # ******************* VALIDATE PHASE ************************
    logging.info('VALID PHASE:')
    results = []
    model.eval()
    for pair in tqdm(valid_gt):
        path0, path1 = pair[:2]
        scene = path0.split('/')[0]
        name0, name1 = path0.split('/')[-1], path1.split('/')[-1]
        if len(pair) >= 5:
            rot0, rot1 = int(pair[2]), int(pair[3])
        else:
            rot0, rot1 = 0, 0

        features0 = valid_data[scene][name0]
        features1 = valid_data[scene][name1]
        inputs = {
            'shape0': {
                'height': torch.tensor(features0['shape']['height']),
                'width': torch.tensor(features0['shape']['width'])},
            'descriptors0': torch.from_numpy(features0['descriptors'].transpose()).unsqueeze(0),
            'keypoints0': torch.from_numpy(features0['keypoints']).unsqueeze(0),
            'scores0': torch.from_numpy(features0['scores']).unsqueeze(0),
            'shape1': {
                'height': torch.tensor(features1['shape']['height']),
                'width': torch.tensor(features1['shape']['width'])},
            'descriptors1': torch.from_numpy(features1['descriptors'].transpose()).unsqueeze(0),
            'keypoints1': torch.from_numpy(features1['keypoints']).unsqueeze(0),
            'scores1': torch.from_numpy(features1['scores']).unsqueeze(0)
        }
        with torch.no_grad():
            pred = model(inputs, is_train=False)
        ###############################################
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        # Estimate the pose and compute the pose error.
        assert len(pair) == 38, 'Pair does not have ground truth info'
        K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
        K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
        T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

        # Update the intrinsics + extrinsics if EXIF rotation was found.
        if rot0 != 0 or rot1 != 0:
            cam0_T_w = np.eye(4)
            cam1_T_w = T_0to1
            if rot0 != 0:
                K0 = rotate_intrinsics(
                    K0,
                    (features0['shape']['height'], features0['shape']['width']), rot0)
                cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
            if rot1 != 0:
                K1 = rotate_intrinsics(
                    K1,
                    (features1['shape']['height'], features1['shape']['width']), rot1)
                cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
            cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
            T_0to1 = cam1_T_cam0

        epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
        correct = epi_errs < 5e-4
        num_correct = np.sum(correct)
        precision = np.mean(correct) if len(correct) > 0 else 0
        matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

        thresh = 1.  # In pixels relative to resized image size.
        ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
        if ret is None:
            err_t, err_R = np.inf, np.inf
        else:
            R, t, inliers = ret
            err_t, err_R = compute_pose_error(T_0to1, R, t)

        # Write the evaluation results to disk.
        out_eval = {'error_t': err_t,
                    'error_R': err_R,
                    'precision': precision,
                    'matching_score': matching_score,
                    'num_correct': num_correct,
                    'epipolar_errors': epi_errs}
        results.append(out_eval)

    pose_errors = []
    precisions = []
    matching_scores = []
    for result in results:
        pose_error = np.maximum(result['error_t'], result['error_R'])
        pose_errors.append(pose_error)
        precisions.append(result['precision'])
        matching_scores.append(result['matching_score'])
    thresholds = [5, 10, 20]
    aucs = pose_auc(pose_errors, thresholds)
    aucs = [100.*yy for yy in aucs]
    prec = 100.*np.mean(precisions)
    ms = 100.*np.mean(matching_scores)
    print('Evaluation Results (mean over {} pairs):'.format(len(valid_gt)))
    print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
        aucs[0], aucs[1], aucs[2], prec, ms))
