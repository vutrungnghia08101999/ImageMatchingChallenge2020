import argparse
import os
import traceback
import math
import logging
from multiprocessing import Pool
from functools import partial
import time

import pandas as pd
import torch
import numpy as np
import cv2
from tqdm import tqdm

from extract_patches.core import extract_patches
from colmap.scripts.python.read_write_model import read_model
from extract_keypoint import get_SIFT_keypoints
from matching import match
from utils import generate_test_set, generate_test_set_csv, load_images


def detect_match(img_dict, image_metadata, device):
    img_kp_dict = {}
    sift = cv2.xfeatures2d.SIFT_create()

    with torch.no_grad():
        for img_id, img in tqdm(img_dict.items()):
            datum = image_metadata[img_id]
            metadata = {"keypoint": datum.xys}
            
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            pred_kps, sizes, response = get_SIFT_keypoints(sift, gray)
        
            try:
                match_ids, dist = match(
                    torch.tensor(datum.xys, dtype=torch.float16).to(device), 
                    torch.tensor(pred_kps, dtype=torch.float16).to(device),
                )
            except:
                logging.info(len(datum.xys), len(pred_kps))
                traceback.print_exc()
                # traceback.print_exception()
                exit()
            
            matched_kps = pred_kps[match_ids]
            matched_sizes = sizes[match_ids]
            metadata["matched_kps"] = matched_kps
            metadata["kp_sizes"] = matched_sizes
            metadata["matching_distance"] = dist
            
            img_kp_dict[img_id] = metadata

    return img_kp_dict


def group_kp_by_3dpoint(points3d, kp_dict, save_path=""):
    points3d_dict = {}
    
    tbar = tqdm(points3d.items())
    for point_id, datum in tbar:
        dist, sizes, loc = [], [], []
        for img_id, point2D_id in zip(datum.image_ids, datum.point2D_idxs):
            dist.append(kp_dict[img_id]["matching_distance"][point2D_id])
            sizes.append(kp_dict[img_id]["kp_sizes"][point2D_id])
            loc.append(kp_dict[img_id]["keypoint"][point2D_id])
        
        dist = np.array(dist)
        sorted_ids = np.argsort(dist)
        
        points3d_dict[point_id] = dict(
            loc     = np.array(loc)[sorted_ids],
            sizes   = np.array(sizes)[sorted_ids],
            dist    = dist[sorted_ids],
            img_ids = datum.image_ids[sorted_ids],
        )
    if save_path:
        np.save(save_path, points3d_dict)
        logging.info(f"matching result saved to {save_path}")
    return points3d_dict


def extract2png(
    points3d_dict, img_dict, patch_per_point, match_thresh, out_folder, img_per_folder=10**4
):
    
    valid_count = 0
    tbar = tqdm(points3d_dict.items())
    for i, (point_id, datum) in enumerate(tbar):
        # filter out 3point that have poor match
        num_patch = min(patch_per_point, len(datum["loc"]))
        dist = np.sqrt(datum["dist"][num_patch-1])
        if match_thresh < dist or num_patch < 2:
            continue
        
        patches = []
        for j, (loc, size, img_id) in enumerate(zip(
            datum["loc"], datum["sizes"], datum["img_ids"]
        )):
            if j == num_patch: break
            kp = cv2.KeyPoint(
                x=loc[0], 
                y=loc[1],
                _size=size,
                _angle=0,
            )
            patch = extract_patches([kp], img_dict[img_id], 32, 12.0)[0]
            patches.append(patch)
        patches = np.concatenate(patches, axis=1)
        
        current_folder = os.path.join(
            out_folder, 
            f"{i//img_per_folder:0>4}", 
            f"thresh_{math.ceil(dist)}"
        )
            
        if not os.path.exists(current_folder):
            os.makedirs(current_folder)
        cv2.imwrite(os.path.join(current_folder, f"{point_id}.png"), patches)
        
        valid_count += 1
        tbar.set_description(f"{valid_count:0>7}")
    logging.info(f"removed {len(points3d_dict) - valid_count} point(s) for poor matching results")


def extract2npy(
    points3d_dict, img_dict, patch_per_point, match_thresh, export_path
):
    rgb_dict = {}
    gray_dict = {}
    valid_count = 0
    tbar = tqdm(points3d_dict.items())
    for point_id, datum in tbar:
        # filter out 3point that have poor match
        num_patch = min(patch_per_point, len(datum["loc"]))
        dist = np.sqrt(datum["dist"][num_patch-1])
        if match_thresh < dist or num_patch < 2:
            continue
        
        gray_patches = []
        rgb_patches = []
        for j, (loc, size, img_id) in enumerate(zip(
            datum["loc"], datum["sizes"], datum["img_ids"]
        )):
            if j == num_patch: break
            kp = cv2.KeyPoint(
                x=loc[0], 
                y=loc[1],
                _size=size,
                _angle=0,
            )
            patch = extract_patches([kp], img_dict[img_id], 32, 12.0)[0]
            rgb_patches.append(patch)
            gray_patches.append(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY))
            
        rgb_dict[point_id] = np.stack(rgb_patches, axis=0)
        gray_dict[point_id] = np.stack(gray_patches, axis=0)
        
        valid_count += 1
        tbar.set_description(f"{valid_count:0>7}")
    logging.info(f"removed {len(points3d_dict) - valid_count} point(s) for poor matching results")

    logging.info("saving patches ....")
    np.save(os.path.join(export_path, 'gray.npy'), gray_dict)
    np.save(os.path.join(export_path, 'rgb.npy'), rgb_dict)
    
    return rgb_dict, gray_dict


def extract2npy_pool(
    points3d_dict, img_dict, patch_per_point, match_thresh, export_path
):
    print(f"processing {len(points3d_dict)} 3d points")
    
    _extract_ = partial(
        extract_,
        img_dict=img_dict,
        patch_per_point=patch_per_point,
        match_thresh=match_thresh
    )
    start = time.perf_counter()
    keys, values = zip(*points3d_dict.items())
    with Pool(5) as p:
        patches_list = p.map(_extract_, values)
    rgb_patches, gray_patches = zip(*patches_list)
    
    gray_dict = dict([(key, patch) for key, patch in zip(keys, gray_patches) if patch is not None])
    rgb_dict = dict([(key, patch) for key, patch in zip(keys, rgb_patches) if patch is not None])
    
    print("Elapsed time: ", time.perf_counter() - start)
    
    print(f"removed {len(points3d_dict) - len(gray_dict)} point(s) for poor matching results")

    print("saving patches ....")
    np.save(os.path.join(export_path, 'gray.npy'), gray_dict)
    np.save(os.path.join(export_path, 'rgb.npy'), rgb_dict)
    ids = list(rgb_dict.keys())
    sorted(ids)
    point_df = pd.DataFrame({"point_id": ids})
    point_df.to_csv(os.path.join(export_path, 'points.csv'), index=False)
    return rgb_dict, gray_dict


def extract_(datum, img_dict, patch_per_point, match_thresh):
    # filter out 3point that have poor match
    num_patch = min(patch_per_point, len(datum["loc"]))
    dist = np.sqrt(datum["dist"][num_patch-1])
    if match_thresh < dist or num_patch < 2:
        return None, None
    
    gray_patches = []
    rgb_patches = []
    for j, (loc, size, img_id) in enumerate(zip(
        datum["loc"], datum["sizes"], datum["img_ids"]
    )):
        if j == num_patch: break
        kp = cv2.KeyPoint(
            x=loc[0], 
            y=loc[1],
            _size=size,
            _angle=0,
        )
        patch = extract_patches([kp], img_dict[img_id], 32, 12.0)[0]
        rgb_patches.append(patch)
        gray_patches.append(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY))
        
    return (np.stack(rgb_patches, axis=0), np.stack(gray_patches, axis=0))


def main():
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            logging.info(f"using {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logging.info("using cpu only")
    except:
        device = torch.device('cpu')
        logging.info("using cpu only")
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_path', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--patch_per_point', type=int, default=2)
    parser.add_argument('--match_thresh', type=float, default=5.0)
    parser.add_argument('--export_path', type=str, default="")
    parser.add_argument('--matching_only', action="store_true")
    parser.add_argument('--num_triplets', type=int, default=50000)
    args = parser.parse_args()

    if not os.path.exists(args.out_path): os.makedirs(args.out_path)
    
    ###### load scene meta data ######
    _, images, points = read_model(path=os.path.join(args.scene_path, "sparse"), ext='.bin')

    logging.info(f"loading images from {args.scene_path}")
    img_dict = load_images(images, os.path.join(args.scene_path, "images"))

    #### detect keypoint and matching ######
    logging.info("detecting keypoint and matching")
    result_path = os.path.join(args.out_path, "matching_result.npy")
    if os.path.exists(result_path):
        point3d_dict = np.load(result_path, allow_pickle=True).item()
        logging.info(f"result {result_path} exist, skip matching")
    else:
        img_kp_dict = detect_match(img_dict, images, device)
        point3d_dict = group_kp_by_3dpoint(points, img_kp_dict, result_path)
    
    ### extract ###
    if not args.matching_only:
        if args.export_path:
            if not os.path.exists(args.export_path): 
                os.makedirs(args.export_path)
            
            if (os.path.exists(os.path.join(args.export_path, 'gray.npy'))
                and os.path.exists(os.path.join(args.export_path, 'rgb.npy'))
            ):
                rgb_dict = np.load(
                    os.path.join(args.export_path, 'rgb.npy'), 
                    allow_pickle=True
                ).item()
                gray_dict = np.load(
                    os.path.join(args.export_path, 'gray.npy'), 
                    allow_pickle=True
                ).item()
                logging.info(f"patches exist, skip extracting")
            else:

                logging.info(f"extract patches and save to {args.export_path}")
                rgb_dict, gray_dict = extract2npy(
                    points3d_dict=point3d_dict, 
                    img_dict=img_dict, 
                    patch_per_point=args.patch_per_point,
                    match_thresh=args.match_thresh,
                    export_path=args.export_path,
                )
                
            points_ids = list(rgb_dict.keys())
            sorted(points_ids)
            point_df = pd.DataFrame({"point_id": points_ids})
            point_df.to_csv(os.path.join(args.export_path, 'points.csv'), index=False)
            
            # rgb_test = generate_test_set(points_ids, args.num_triplets, rgb_dict)
            # np.save(os.path.join(
            #     args.export_path, 
            #     f'{args.num_triplets}_{args.num_triplets}_rgb.npy'
            # ), rgb_test)
            
            # gray_test = generate_test_set(points_ids, args.num_triplets, gray_dict)
            # np.save(os.path.join(
            #     args.export_path, 
            #     f'{args.num_triplets}_{args.num_triplets}_gray.npy'
            # ), gray_test)
            
            generate_test_set_csv(points_ids, os.path.join(args.export_path, 'val.csv'))
        else:
            
            extract2png(
                points3d_dict=point3d_dict, 
                img_dict=img_dict, 
                patch_per_point=args.patch_per_point,
                match_thresh=args.match_thresh,
                out_folder=os.path.join(args.out_path, "png"),
            )

if __name__ == "__main__":
    logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    main()
