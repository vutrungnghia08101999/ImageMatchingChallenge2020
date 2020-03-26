import logging
import math
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

INPUT = 'raw/buckingham_palace'
OUTPUT = 'rgb/buckingham_palace'


def concatenate_patches(patches: list):
    assert len(patches) <= 256
    cache = patches.copy()
    n = len(cache)
    for i in range(256 - n):
        cache.append(np.zeros((64, 64, 3), np.uint8) + 255)
    
    rows = []
    for row in range(16):
        rows.append(np.concatenate(cache[row * 16: row * 16 + 16], axis=1))
    return np.concatenate(rows, axis=0)


# sort .pt file
files = []
for filename in list(os.listdir(INPUT)):
    start = int(filename.split('_')[0])
    files.append((start, filename))
files.sort(key=lambda tup: tup[0])
files = [x[1] for x in files]


# read points and patches from list of .pts
point_ids = []
patches = []
for file in tqdm(files):
    try:
        s = torch.load(os.path.join(INPUT, file))
        for dic in s:
            point_ids.append(dic['point_id'])
            patches.append(dic['patch'])
    except FileNotFoundError as e:
        print(e)

#write info and batch image
with open(os.path.join(OUTPUT, 'info.txt'), 'w') as f:
    for item in point_ids:
        f.write(f"{item}\n")

n_batches = math.ceil(len(patches)/256)
for index in tqdm(range(n_batches)):
    image = concatenate_patches(patches[index * 256: index * 256 + 256])
    image = Image.fromarray(image)
    image.save(os.path.join(OUTPUT, f'{str(index).zfill(6)}.bmp'))

logging.info(f'input: {INPUT}')
logging.info(f'output: {OUTPUT}')
logging.info(f'unique_3d_ids: {len(pd.Series(point_ids).unique())}')
logging.info(f'n_patches: {len(point_ids)}')
logging.info('Completed\n')
