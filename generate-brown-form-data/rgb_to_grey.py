import os
from PIL import Image
import shutil
from tqdm import tqdm

import numpy as np

INPUT = 'rgb/buckingham_palace'
OUTPUT = 'grey/buckingham_palace'

shutil.copyfile(os.path.join(INPUT, 'info.txt'), os.path.join(OUTPUT, 'info.txt'))
shutil.copyfile(os.path.join(INPUT, '50000_50000.txt'), os.path.join(OUTPUT, '50000_50000.txt'))

files = [x for x in list(os.listdir(INPUT)) if x.endswith('.bmp')]

for file in tqdm(files):
    image = Image.open(os.path.join(INPUT, file))
    image = np.array(image)[:, :, 0]
    image = Image.fromarray(image)
    image.save(os.path.join(OUTPUT, file))

