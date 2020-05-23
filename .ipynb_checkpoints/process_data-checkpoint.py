import argparse
import itertools
import logging
import os
import random
from tqdm import tqdm

import numpy as np
from PIL import Image

from colmap.scripts.python.read_write_model import read_points3d_binary
logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='/media/vutrungnghia/New Volume/P2-ImageMatchingChallenge/dataset/superglue/input')
parser.add_argument('--output', type=str, default='/media/vutrungnghia/New Volume/P2-ImageMatchingChallenge/dataset/superglue/output')
parser.add_argument('--scene', type=str, default='reichstag')
parser.add_argument('--iou_thresold', type=float, default=0.1)
args = parser.parse_args()

logging.info('\n======== SUPERGLUE - DATA PREPROCESSING ========\n')
logging.info(args._get_kwargs())

INPUT = os.path.join(args.input, args.scene)
OUTPUT = os.path.join(args.output, f'{args.scene}.npy')

# load data from bin files
points = read_points3d_binary(os.path.join(INPUT, 'dense/sparse/points3D.bin'))
local_features = np.load(os.path.join(INPUT, 'local_features.npy'), allow_pickle=True).item()

logging.info(f'No.points: {len(points)}')
logging.info(f'No.images: {len(local_features)}')
logging.info(f'max no.pairs: {int(len(local_features) * (len(local_features) - 1)/2)}')

# count number of matches for every possible pair and filter base on iou_thresold
all_pairs = {}
for point_id in tqdm(points.keys()):
    point = points[point_id]
    if len(point.image_ids) < 2:
        continue
    pairs = itertools.combinations(sorted(point.image_ids), 2)
    pairs = list(pairs)
    for pair in pairs:
        if pair[0] == pair[1]:  # do not add pair of the same image
            continue
        if pair not in all_pairs:
            all_pairs[pair] = {'intersection': 0}
        all_pairs[pair]['intersection'] += 1

for (u, v) in all_pairs.keys():
    u_kpts = local_features[u]['xys'].shape[0]
    v_kpts = local_features[v]['xys'].shape[0]
    all_pairs[(u, v)]['union'] = u_kpts + v_kpts - all_pairs[(u, v)]['intersection']
    all_pairs[(u, v)]['iou'] = all_pairs[(u, v)]['intersection'] / all_pairs[(u, v)]['union']

dataset = {}
for k, v in all_pairs.items():
    if v['iou'] > args.iou_thresold:
        dataset[k] = {'matches': []}

logging.info(f'No.pair filter/all: {len(dataset)}/{len(all_pairs)}')

# generate list of matched index keypoints for each pairs after filtering
for point_id in tqdm(points.keys()):
    point = points[point_id]
    if len(point.image_ids) < 2:
        continue
    indices = {}
    for i in range(len(point.image_ids)):
        indices[point.image_ids[i]] = point.point2D_idxs[i]
    pairs = itertools.combinations(sorted(point.image_ids), 2)
    pairs = list(pairs)
    for pair in pairs:
        if pair[0] == pair[1]:  # do not add pair of the same image
            continue
        if pair not in dataset:  # pair is filter out from previous step
            continue
        dataset[pair]['matches'].append((indices[pair[0]], indices[pair[1]]))

for (img0, img1) in tqdm(dataset.keys()):
    keypoints0, keypoints1 = local_features[img0]['xys'], local_features[img1]['xys']
    descriptors0, descriptors1 = local_features[img0]['descriptors'], local_features[img1]['descriptors']
    scores0, scores1 = local_features[img0]['confident_scores'], local_features[img1]['confident_scores']
    points3d_IDs0, points3d_IDs1 = local_features[img0]['points3D_ids'], local_features[img1]['points3D_ids']
    shape0 = {
        'width': local_features[img0]['width'],
        'height': local_features[img0]['height']}
    shape1 = {
        'width': local_features[img1]['width'],
        'height': local_features[img1]['height']}
    dataset[(img0, img1)]['keypoints'] = (keypoints0, keypoints1)
    dataset[(img0, img1)]['descriptors'] = (descriptors0, descriptors1)
    dataset[(img0, img1)]['scores'] = (scores0, scores1)
    dataset[(img0, img1)]['3dpoints'] = (points3d_IDs0, points3d_IDs1)
    dataset[(img0, img1)]['shape'] = (shape0, shape1)
    dataset[(img0, img1)]['name'] = (local_features[img0]['name'], local_features[img1]['name'])

# statistics of data
matches = []
keypoints0 = []
keypoints1 = []
points3d_IDs0 = []
points3d_IDs1 = []
has_descriptors0 = []
has_descriptors1 = []
counter = 0
for k, v in tqdm(dataset.items()):
    matches.append(len(v['matches']))
    keypoints0.append(v['keypoints'][0].shape[0])
    keypoints1.append(v['keypoints'][1].shape[0])
    points3d_IDs0.append(sum(v['3dpoints'][0] != -1))
    points3d_IDs1.append(sum(v['3dpoints'][1] != -1))
    has_descriptors0.append(sum(~np.isnan(v['descriptors'][0][:, 0])))
    has_descriptors1.append(sum(~np.isnan(v['descriptors'][1][:, 0])))
    counter += 1
    if counter == 2000:
        break

logging.info(f'avg matches: {sum(matches)/len(matches)}')
logging.info(f'avg keypoints0: {sum(keypoints0)/len(keypoints0)}')
logging.info(f'avg keypoints1: {sum(keypoints1)/len(keypoints1)}')
logging.info(f'avg points3d_IDs0: {sum(points3d_IDs0)/len(points3d_IDs0)}')
logging.info(f'avg points3d_IDs1: {sum(points3d_IDs1)/len(points3d_IDs1)}')
logging.info(f'avg has_descriptors0: {sum(has_descriptors0)/len(has_descriptors0)}')
logging.info(f'avg has_descriptors1: {sum(has_descriptors1)/len(has_descriptors1)}')

# save data
np.save(OUTPUT, dataset)
logging.info(f'Saved data at: {OUTPUT}')
logging.info('Completed\n')

# pair = dataset[(1, 2)]
# this function is used to create groundtruth, filtered keypoints, descriptors and scores on the fly during training
def transform_and_filter(pair: dict) -> dict:
    """get groundtruth matrix for a pair of images

    Arguments:
        pair {dict}:
            matches: [(idx1, idx2), .... ]
            keypoints: (N x 2, M x 2)
            descriptors: (N x 128, M x 128)
            scores: (N, M)
            3dpoints: (N, M)
            shape: ({width: , height: }, {width: , height: })
            name: (str, str)
    Returns:
        np.array -- [description]
    """
    assert pair['keypoints'][0].shape[0] == pair['descriptors'][0].shape[0]
    assert pair['keypoints'][0].shape[0] == pair['scores'][0].shape[0]
    assert pair['keypoints'][0].shape[0] == pair['3dpoints'][0].shape[0]

    assert pair['keypoints'][1].shape[0] == pair['descriptors'][1].shape[0]
    assert pair['keypoints'][1].shape[0] == pair['scores'][1].shape[0]
    assert pair['keypoints'][1].shape[0] == pair['3dpoints'][1].shape[0]

    # get keypoints, descriptors and scores base on filter of has 3d points and descriptors
    has_3d_points0 = pair['3dpoints'][0] != -1
    has_descriptors0 = ~np.isnan(pair['descriptors'][0][:, 0])
    filter0 = has_3d_points0 * has_descriptors0
    keypoints0 = pair['keypoints'][0][filter0]
    descriptors0 = pair['descriptors'][0][filter0]
    scores0 = pair['scores'][0][filter0]

    has_3d_points1 = pair['3dpoints'][1] != -1
    has_descriptors1 = ~np.isnan(pair['descriptors'][1][:, 0])
    filter1 = has_3d_points1 * has_descriptors1
    keypoints1 = pair['keypoints'][1][filter1]
    descriptors1 = pair['descriptors'][1][filter1]
    scores1 = pair['scores'][1][filter1]

    # construct the groundtruth matrix for keypoints after filtering
    height, width = pair['keypoints'][0].shape[0] + 1, pair['keypoints'][1].shape[0] + 1
    groundtruth = np.zeros((height, width))
    right_append = np.ones(height)
    bottom_append = np.ones(width)

    for kp1_idx, kp2_idx in pair['matches']:
        groundtruth[kp1_idx][kp2_idx] = 1
        right_append[kp1_idx] = 0
        bottom_append[kp2_idx] = 0

    groundtruth[:, width - 1] += right_append
    groundtruth[height - 1, :] += bottom_append
    groundtruth[height - 1][width - 1] = 0

    filter0 = np.append(filter0, True)
    filter1 = np.append(filter1, True)
    groundtruth = groundtruth[filter0, :]
    groundtruth = groundtruth[:, filter1]

    return {
        'keypoints': (keypoints0, keypoints1),
        'descriptors': (descriptors0, descriptors1),
        'scores': (scores0, scores1),
        'shape': pair['shape'],
        'name': pair['name'],
        'groundtruth': groundtruth
    }

# import matplotlib.pyplot as plt
# s = transform_and_filter(dataset[list(dataset.keys())[0]])
# image0 = Image.open(os.path.join('/media/vutrungnghia/New Volume/P2-ImageMatchingChallenge/dataset/challenge-dataset/valid/reichstag/dense/images', s['name'][0]))
# image1 = Image.open(os.path.join('/media/vutrungnghia/New Volume/P2-ImageMatchingChallenge/dataset/challenge-dataset/valid/reichstag/dense/images', s['name'][1]))
# a = np.argwhere(s['groundtruth'][:-1, :-1] == 1)
# kps0 = s['keypoints'][0]
# kps1 = s['keypoints'][1]

# p = 9
# kp0 = a[p][0]
# kp1 = a[p][1]

# plt.imshow(image0)
# plt.scatter(kps0[kp0][0], kps0[kp0][1])


# plt.imshow(image1)
# plt.scatter(kps1[kp1][0], kps1[kp1][1])

#
# s = transform_and_filter(dataset[list(dataset.keys())[0]])
# a = np.argwhere(s['groundtruth'][:-1, :-1] == 1)
# descriptors0 = s['descriptors'][0]
# descriptors1 = s['descriptors'][1]

# a1 = []
# for u in a:
#     v1 = descriptors0[u[0]]
#     v2 = descriptors1[u[1]]
#     a1.append(np.sqrt(sum((v1 - v2) ** 2)))

# a2 = []
# for i in range(500):
#     v1 = descriptors0[i]
#     v2 = descriptors1[i]
#     a2.append(np.sqrt(sum((v1 - v2) ** 2)))

# f1 = sum(a1)/len(a1)
# f2 = sum(a2)/len(a2)