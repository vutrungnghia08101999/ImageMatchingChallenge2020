import os
from PIL import Image
import shutil
from tqdm import tqdm

import numpy as np

############### PARAMETERS ##############
INPUT = 'rgb/reichstag'
OUTPUT = 'gray/reichstag'
INFO_FILE = 'info.txt'
MATCHES_FILE = '50000_50000.txt'
IMAGE_EXT = 'bmp'
#########################################

shutil.copyfile(os.path.join(INPUT, INFO_FILE), os.path.join(OUTPUT, INFO_FILE))
shutil.copyfile(os.path.join(INPUT, MATCHES_FILE), os.path.join(OUTPUT, MATCHES_FILE))

files = [x for x in list(os.listdir(INPUT)) if x.endswith(IMAGE_EXT)]

for file in tqdm(files):
    image = Image.open(os.path.join(INPUT, file))
    image = np.array(image)[:, :, 0]
    image = Image.fromarray(image)
    if os.path.isfile(os.path.join(OUTPUT, file)):
        raise RuntimeError(f'{os.path.join(OUTPUT, file)} is already exist')
    image.save(os.path.join(OUTPUT, file))
