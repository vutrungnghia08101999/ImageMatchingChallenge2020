import logging
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

############### PARAMETERS ##############
INPUT = 'gray/reichstag'
INFO_FILE = 'info.txt'
MATCHES_FILE = '50000_50000.txt'
IMAGE_EXT = 'bmp'
N_PATCHES = 35262
COMPRESS_FILE = 'gray/reichstag.pt'
#########################################

logging.info('=========================================================')
logging.info('\nCOMPRESS GRAY DATA\n')
logging.info(f'input: {INPUT}')
logging.info(f'info_file: {INFO_FILE}')
logging.info(f'matches_files: {MATCHES_FILE}')
logging.info(f'n_patches: {N_PATCHES}')
logging.info(f'compress_file: {COMPRESS_FILE}')


def read_image_file(data_dir, image_ext, n):
    """Return a Tensor containing the patches
    """

    def PIL2array(_img):
        """Convert PIL image type to numpy 2D array
        """
        return np.array(_img.getdata(), dtype=np.uint8).reshape(64, 64)

    def find_files(_data_dir, _image_ext):
        """Return a list with the file names of the images containing the patches
        """
        files = []
        # find those files with the specified extension
        for file_dir in os.listdir(_data_dir):
            if file_dir.endswith(_image_ext):
                files.append(os.path.join(_data_dir, file_dir))
        return sorted(files)  # sort files in ascend order to keep relations

    patches = []
    list_files = find_files(data_dir, image_ext)

    for fpath in tqdm(list_files):
        img = Image.open(fpath)
        for y in range(0, 1024, 64):
            for x in range(0, 1024, 64):
                patch = img.crop((x, y, x + 64, y + 64))
                patches.append(PIL2array(patch))
    return torch.ByteTensor(np.array(patches[:n]))


def read_info_file(data_dir, info_file):
    """Return a Tensor containing the list of labels
       Read the file and keep only the ID of the 3D point.
    """
    labels = []
    with open(os.path.join(data_dir, info_file), 'r') as f:
        labels = [int(line.split()[0]) for line in f]
    return torch.LongTensor(labels)


def read_matches_files(data_dir, matches_file):
    """Return a Tensor containing the ground truth matches
       Read the file and keep only 3D point ID.
       Matches are represented with a 1, non matches with a 0.
    """
    matches = []
    with open(os.path.join(data_dir, matches_file), 'r') as f:
        for line in f:
            line_split = line.split()
            matches.append([int(line_split[0]), int(line_split[3]),
                            int(line_split[1] == line_split[4])])
    return torch.LongTensor(matches)

dataset = (
    read_image_file(INPUT, IMAGE_EXT, N_PATCHES),
    read_info_file(INPUT, INFO_FILE),
    read_matches_files(INPUT, MATCHES_FILE)
)

if os.path.isfile(COMPRESS_FILE):
    raise RuntimeError(f'{COMPRESS_FILE} is already exist.')
with open(COMPRESS_FILE, 'wb') as f:
    torch.save(dataset, f)

logging.info('Completed\n')
