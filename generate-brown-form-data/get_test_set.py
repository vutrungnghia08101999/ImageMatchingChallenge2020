import logging
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

INPUT = 'rgb/buckingham_palace'
INFO_FILE = 'info.txt'
IMAGE_EXTENSION = 'bmp'
N_POS_NEG = 50000
BATCH_SIZE = 500 # generate 10000 pos pairs and 10000 neg pairs at a time (id is unique)

logging.info(f'input: {INPUT}')
logging.info(f'n_pos: {N_POS_NEG} - n_neg: {N_POS_NEG}')
logging.info(f'generated_batch_size: {BATCH_SIZE}')

text = open(os.path.join(INPUT, INFO_FILE)).read()
text = text.replace(' ', '')
labels = [int(x) for x in text.split('\n') if x != '']

def generate_triplets(labels, num_triplets):
# labels = torch.tensor(labels)
# num_triplets = N_POS_NEG

    def create_indices(_labels):
        inds = dict()
        for idx, ind in enumerate(_labels):
            if ind not in inds:
                inds[ind] = []
            inds[ind].append(idx)
        return inds

    triplets = []
    indices = create_indices(labels.numpy())
    unique_labels = np.unique(labels.numpy())
    n_classes = unique_labels.shape[0]
    # add only unique indices in batch
    already_idxs = set()

    for x in tqdm(range(num_triplets)):
        x = 0
        if len(already_idxs) >= BATCH_SIZE:
            already_idxs = set()
        c1 = unique_labels[np.random.randint(0, n_classes)]
        while c1 in already_idxs or len(indices[c1]) < 2:
            c1 = unique_labels[np.random.randint(0, n_classes)]
        already_idxs.add(c1)
        c2 = unique_labels[np.random.randint(0, n_classes)]
        while c1 == c2:
            c2 = unique_labels[np.random.randint(0, n_classes)]
        if len(indices[c1]) == 2:  # hack to speed up process
            n1, n2 = 0, 1
        else:
            n1 = np.random.randint(0, len(indices[c1]))
            n2 = np.random.randint(0, len(indices[c1]))
            while n1 == n2:
                n2 = np.random.randint(0, len(indices[c1]))
        n3 = np.random.randint(0, len(indices[c2]))
        # triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
        triplets.append([indices[c1][n1], c1, -1, indices[c1][n2], c1, -1])
        triplets.append([indices[c1][n1], c1, -1, indices[c2][n3], c2, -1])
    return torch.LongTensor(np.array(triplets))

test_set = generate_triplets(torch.tensor(labels), N_POS_NEG)
test_set_file = os.path.join(INPUT, f'{N_POS_NEG}_{N_POS_NEG}.txt')
np.savetxt(test_set_file, test_set.numpy(), fmt='%d')
logging.info('Completed\n')
