from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
from torchvision.transforms import ToPILImage

from models.matching import Matching
from models.superpoint import SuperPoint
from models.superglue import SuperGlue
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)


# if __name__ == '__main__':
parser = argparse.ArgumentParser(
    description='Image pair matching and pose evaluation with SuperGlue',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--input_pairs', type=str, default='assets/scannet_sample_pairs_with_gt.txt',
    help='Path to the list of image pairs')
parser.add_argument(
    '--input_dir', type=str, default='assets/scannet_sample_images/',
    help='Path to the directory that contains the images')
parser.add_argument(
    '--output_dir', type=str, default='dump_match_pairs/',
    help='Path to the directory in which the .npz results and optionally,'
            'the visualization images are written')

parser.add_argument(
    '--resize', type=int, nargs='+', default=[640, 480],
    help='Resize the input image before running inference. If two numbers, '
            'resize to the exact dimensions, if one number, resize the max '
            'dimension, if -1, do not resize')
parser.add_argument(
    '--resize_float', action='store_true',
    help='Resize the image after casting uint8 to float')

parser.add_argument(
    '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
    help='SuperGlue weights')
parser.add_argument(
    '--max_keypoints', type=int, default=1024,
    help='Maximum number of keypoints detected by Superpoint'
            ' (\'-1\' keeps all keypoints)')
parser.add_argument(
    '--keypoint_threshold', type=float, default=0.005,
    help='SuperPoint keypoint detector confidence threshold')
parser.add_argument(
    '--nms_radius', type=int, default=4,
    help='SuperPoint Non Maximum Suppression (NMS) radius'
    ' (Must be positive)')
parser.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations performed by SuperGlue')
parser.add_argument(
    '--match_threshold', type=float, default=0.2,
    help='SuperGlue match threshold')

parser.add_argument(
    '--viz', action='store_true',
    help='Visualize the matches and dump the plots')
parser.add_argument(
    '--eval', action='store_true',
    help='Perform the evaluation'
            ' (requires ground truth pose and intrinsics)')
parser.add_argument(
    '--fast_viz', action='store_true',
    help='Use faster image visualization with OpenCV instead of Matplotlib')
parser.add_argument(
    '--cache', action='store_true',
    help='Skip the pair if output .npz files are already found')
parser.add_argument(
    '--show_keypoints', action='store_true',
    help='Plot the keypoints in addition to the matches')
parser.add_argument(
    '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
    help='Visualization file extension. Use pdf for highest-quality.')
parser.add_argument(
    '--opencv_display', action='store_true',
    help='Visualize via OpenCV before saving output images')
parser.add_argument(
    '--shuffle', action='store_true',
    help='Shuffle ordering of pairs before processing')
parser.add_argument(
    '--force_cpu', action='store_true',
    help='Force pytorch to run in CPU mode.')

args = parser.parse_args(['--eval'])
print(args)

assert not (args.opencv_display and not args.viz), 'Must use --viz with --opencv_display'
assert not (args.opencv_display and not args.fast_viz), 'Cannot use --opencv_display without --fast_viz'
assert not (args.fast_viz and not args.viz), 'Must use --viz with --fast_viz'
assert not (args.fast_viz and args.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

print(f'Will resize to {args.resize[0]}x{args.resize[1]} (WxH)')

with open(args.input_pairs, 'r') as f:
    pairs = [l.split() for l in f.readlines()]

# Load the SuperPoint and SuperGlue models.
device = 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu'
print('Running inference on device \"{}\"'.format(device))
config = {
    'superpoint': {
        'nms_radius': args.nms_radius,
        'keypoint_threshold': args.keypoint_threshold,
        'max_keypoints': args.max_keypoints
    },
    'superglue': {
        'weights': args.superglue,
        'sinkhorn_iterations': args.sinkhorn_iterations,
        'match_threshold': args.match_threshold,
    }
}
#########################################################
superpoint = SuperPoint(config['superpoint']).to(device)
superglue = SuperGlue(config['superglue']).to(device)

pair = pairs[10]
name0, name1 = pair[:2]
stem0, stem1 = Path(name0).stem, Path(name1).stem

if len(pair) >= 5:
    rot0, rot1 = int(pair[2]), int(pair[3])
else:
    rot0, rot1 = 0, 0
input_dir = Path(args.input_dir)
image0, inp0, scales0 = read_image(
    input_dir / name0, device, args.resize, rot0, args.resize_float)
image1, inp1, scales1 = read_image(
    input_dir / name1, device, args.resize, rot1, args.resize_float)

image0_superpoint_out = superpoint({'image': inp0})
image1_superpoint_out = superpoint({'image': inp1})

superglue_inputs = {
    'image0_shape': ToPILImage()(inp0.squeeze()).size,
    'descriptors0': torch.stack(image0_superpoint_out['descriptors']),
    'keypoints0': torch.stack(image0_superpoint_out['keypoints']),
    'scores0': torch.stack(image0_superpoint_out['scores']),

    'image1_shape': ToPILImage()(inp1.squeeze()).size,
    'descriptors1': torch.stack(image1_superpoint_out['descriptors']),
    'keypoints1': torch.stack(image1_superpoint_out['keypoints']),
    'scores1': torch.stack(image1_superpoint_out['scores'])
}

superglue_out = superglue(superglue_inputs)

#########################################################

matching = Matching(config).eval().to(device)

# Create the output directories if they do not exist already.
input_dir = Path(args.input_dir)
print('Looking for data in directory \"{}\"'.format(input_dir))
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)
print('Will write matches to directory \"{}\"'.format(output_dir))
if args.eval:
    print('Will write evaluation results', 'to directory \"{}\"'.format(output_dir))
if args.viz:
    print('Will write visualization images to', 'directory \"{}\"'.format(output_dir))

timer = AverageTimer(newline=True)

for i, pair in enumerate(pairs):
    pair = pairs[10]
    name0, name1 = pair[:2]
    print(name0, name1)
    stem0, stem1 = Path(name0).stem, Path(name1).stem
    matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
    eval_path = output_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
    viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, args.viz_extension)
    viz_eval_path = output_dir / \
        '{}_{}_evaluation.{}'.format(stem0, stem1, args.viz_extension)

    # Handle --cache logic.
    do_match = True
    do_eval = args.eval
    do_viz = args.viz
    do_viz_eval = args.eval and args.viz

    # If a rotation integer is provided (e.g. from EXIF data), use it:
    if len(pair) >= 5:
        rot0, rot1 = int(pair[2]), int(pair[3])
    else:
        rot0, rot1 = 0, 0

    # Load the image pair.
    image0, inp0, scales0 = read_image(
        input_dir / name0, device, args.resize, rot0, args.resize_float)
    image1, inp1, scales1 = read_image(
        input_dir / name1, device, args.resize, rot1, args.resize_float)

    if do_match:
        # Perform the matching.
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        timer.update('matcher')

        # Write the matches to disk.
        out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                        'matches': matches, 'match_confidence': conf}
        np.savez(str(matches_path), **out_matches)

    # Keep the matching keypoints.
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]

    if do_eval:
        # Estimate the pose and compute the pose error.
        assert len(pair) == 38, 'Pair does not have ground truth info'
        K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
        K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
        T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

        # Scale the intrinsics to resized image.
        K0 = scale_intrinsics(K0, scales0)
        K1 = scale_intrinsics(K1, scales1)

        # Update the intrinsics + extrinsics if EXIF rotation was found.
        if rot0 != 0 or rot1 != 0:
            cam0_T_w = np.eye(4)
            cam1_T_w = T_0to1
            if rot0 != 0:
                K0 = rotate_intrinsics(K0, image0.shape, rot0)
                cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
            if rot1 != 0:
                K1 = rotate_intrinsics(K1, image1.shape, rot1)
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
        np.savez(str(eval_path), **out_eval)

if args.eval:
    # Collate the results into a final table and print to terminal.
    pose_errors = []
    precisions = []
    matching_scores = []
    for pair in pairs:
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        eval_path = output_dir / \
            '{}_{}_evaluation.npz'.format(stem0, stem1)
        results = np.load(eval_path)
        pose_error = np.maximum(results['error_t'], results['error_R'])
        pose_errors.append(pose_error)
        precisions.append(results['precision'])
        matching_scores.append(results['matching_score'])
    thresholds = [5, 10, 20]
    aucs = pose_auc(pose_errors, thresholds)
    aucs = [100.*yy for yy in aucs]
    prec = 100.*np.mean(precisions)
    ms = 100.*np.mean(matching_scores)
    print('Evaluation Results (mean over {} pairs):'.format(len(pairs)))
    print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
        aucs[0], aucs[1], aucs[2], prec, ms))
