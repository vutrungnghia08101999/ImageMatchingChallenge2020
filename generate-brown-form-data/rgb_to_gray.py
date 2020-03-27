import logging
import os
from PIL import Image
import shutil
from tqdm import tqdm

import numpy as np

logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

############### PARAMETERS ##############
INPUT = 'rgb/palace_of_westminster'
OUTPUT = 'gray/palace_of_westminster'

INFO_FILE = 'info.txt'
MATCHES_FILE = '50000_50000.txt'
IMAGE_EXT = 'bmp'
#########################################
if not os.path.isdir(OUTPUT):
    os.makedirs(OUTPUT, exist_ok=False)

logging.info('=========================================================')
logging.info('\nRGB TO GRAY\n')
logging.info(f'input: {INPUT}')
logging.info(f'output: {OUTPUT}')
logging.info(f'info_file: {INFO_FILE}')
logging.info(f'matches_files: {MATCHES_FILE}')

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

logging.info('Completed')
