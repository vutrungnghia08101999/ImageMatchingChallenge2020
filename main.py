import argparse
import datetime
import logging
import os
import random
from tqdm import tqdm
import cv2

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

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--root', type=str,
                    default='/home/hieu123/nghia/dataset/train_data')
parser.add_argument('--train_scenes', type=str,
                    default='reichstag')
parser.add_argument('--sinkhorn_iterations', type=int, default=20,
                    help='Number of Sinkhorn iterations performed by SuperGlue')
parser.add_argument('--match_threshold', type=float, default=0.2,
                    help='SuperGlue match threshold')
parser.add_argument('--weights', type=str,
                    default='')
parser.add_argument('--valid_data', type=str,
                    default='/home/hieu123/nghia/dataset/test_data/val_2048.npy')
parser.add_argument('--valid_gt', type=str,
                    default='valid/yfcc_sample_pairs_with_gt.txt')
parser.add_argument('--models_folder', type=str,
                    default='/home/hieu123/nghia/models')
parser.add_argument('--n_epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()

logging.basicConfig(filename=f'logs/{args.exp_name}.txt',
                    filemode='w',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
# logging.basicConfig(level=logging.INFO)

logging.info(f'\n========= IMAGE MATCHING CHALLENGE 2020 ==========\n')
logging.info(args._get_kwargs())

# creat train dataset and dataloader, load valid data and valid gt
valid_data = np.load(args.valid_data, allow_pickle=True).item()
with open(args.valid_gt, 'r') as f:
    valid_gt = [l.split() for l in f.readlines()]
logging.info(f'VALID_SCENES: {valid_data.keys()}')
for k, v in valid_data.items():
    logging.info(f'  {k}: {len(v)}')
logging.info(f'No.gt_pairs: {len(valid_gt)}')

# create superglue model and load checkpoint if exist
model = SuperGlue({'sinkhorn_iterations': args.sinkhorn_iterations,
                   'match_threshold': args.match_threshold})
if torch.cuda.is_available():
    model = model.cuda()
start_epoch = 0
if args.weights:
    logging.info(f'Load model: {args.weights}')
    checkpoint = torch.load(args.weights)
    model.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch']

# create optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             betas=(0.9, 0.999))

for epoch in range(start_epoch + 1, args.n_epochs):
    logging.info(f'EPOCH: {epoch}/{args.n_epochs}')
    # ******************* TRAINING PHASE ***********************
    logging.info(f'TRAINING PHASE:')
    dataset = SuperGlueDataset(args.root, args.train_scenes.split(','))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    logging.info(f'Generated {len(dataloader)} samples')
    model.train()
    avg_loss = AverageMeter()
    for pair in tqdm(dataloader):
        groundtruth = pair['groundtruth']
        inputs = {
            'shape0': pair['shape'][0],
            'descriptors0': pair['descriptors'][0].cuda(),
            'keypoints0': pair['keypoints'][0].cuda(),
            'scores0': pair['scores'][0].cuda(),
            'shape1': pair['shape'][1],
            'descriptors1': pair['descriptors'][1].cuda(),
            'keypoints1': pair['keypoints'][1].cuda(),
            'scores1': pair['scores'][1].cuda()
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
    x = datetime.datetime.now()
    time = x.strftime("%y-%m-%d_%H:%M:%S")
    model_checkpoint = os.path.join(args.models_folder, f'checkpoint_{time}_{epoch}.pth')
    torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, model_checkpoint)
    logging.info(f'{model_checkpoint}')
    # ******************* VALIDATE TRAIN PHASE **********************
    logging.info('VALID TRAIN PHASE:')
    model.eval()
    avg_keypoints = AverageMeter()
    avg_matches = AverageMeter()
    avg_pred_positive = AverageMeter()
    avg_true_positive = AverageMeter()
    for pair in tqdm(dataloader):
        groundtruth = pair['groundtruth']
        inputs = {
            'shape0': pair['shape'][0],
            'descriptors0': pair['descriptors'][0].cuda(),
            'keypoints0': pair['keypoints'][0].cuda(),
            'scores0': pair['scores'][0].cuda(),
            'shape1': pair['shape'][1],
            'descriptors1': pair['descriptors'][1].cuda(),
            'keypoints1': pair['keypoints'][1].cuda(),
            'scores1': pair['scores'][1].cuda()
        }
        with torch.no_grad():
            pred = model(inputs, is_train=False)
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        matches = pred['matches0']

        indices_img0 = np.argwhere(matches != -1)
        indices_img1 = matches[indices_img0]
        links = np.stack([indices_img0, indices_img1], axis=1)
        links = links[:, :, 0]

        groundtruth = groundtruth.squeeze()[:-1, :-1].numpy().astype(np.int)
        pred_matrix = np.zeros(groundtruth.shape)

        for row, col in links:
            pred_matrix[row][col] = 1
        pred_matrix = pred_matrix.astype(np.int)

        avg_keypoints.update(sum(groundtruth.shape) / len(groundtruth.shape))
        avg_matches.update(groundtruth.sum())
        avg_pred_positive.update(pred_matrix.sum())
        true_positive = (pred_matrix == groundtruth) * groundtruth
        avg_true_positive.update(true_positive.sum())
    logging.info(f'avg_keypoints: {avg_keypoints.avg}')
    logging.info(f'avg_matches: {avg_matches.avg}')
    logging.info(f'avg_pred_positive: {avg_pred_positive.avg}')
    logging.info(f'avg_true_positive: {avg_true_positive.avg}')
    
    # ******************* VALIDATE PHASE ************************
    logging.info('VALID PHASE:')
    results = []
    model.eval()

    summary_result = {}
    for scene in valid_data.keys():
        summary_result[scene] = {'avg_keypoints': AverageMeter(),
                                 'avg_pred_positive': AverageMeter(),
                                 'avg_true_positive': AverageMeter()}
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
        summary_result[scene]['avg_keypoints'].update(len(kpts0))
        summary_result[scene]['avg_pred_positive'].update(len(correct))
        summary_result[scene]['avg_true_positive'].update(num_correct)
        
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
    for scene, v in summary_result.items():
        v1 = v['avg_keypoints'].avg
        v2 = v['avg_pred_positive'].avg
        v3 = v['avg_true_positive'].avg
        logging.info(f'{scene}')
        logging.info(f'  avg_keypoints: {v1}')
        logging.info(f'  avg_pred_positive: {v2}')
        logging.info(f'  avg_true_positive: {v3}')
    logging.info('Evaluation Results (mean over {} pairs):'.format(len(valid_gt)))
    logging.info('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
    logging.info('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
        aucs[0], aucs[1], aucs[2], prec, ms))
