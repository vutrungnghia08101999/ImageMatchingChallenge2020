import os
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn


class TripletPhotoTour(Dataset):
    def __init__(self, root, name, n_triplets:int, fliprot:bool, train=True, batch_size = None, load_random_triplets = False, transform=None): 
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root
        self.name = name
        self.data_dir = os.path.join(self.root, name)
        self.data_file = os.path.join(self.root, '{}.pt'.format(name))
        self.train = train
        
        if not self._check_datafile_exists():
            raise RuntimeError('Dataset not found.')

        self.data, self.labels, self.matches = torch.load(self.data_file)
        
        self.transform = transform
        self.out_triplets = load_random_triplets
        self.n_triplets = n_triplets
        self.batch_size = batch_size
        self.fliprot = fliprot
        
        if self.train:
            print(f'Generating {self.n_triplets} triplets')
            self.triplets = self.generate_triplets(self.labels, self.n_triplets)

    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img

        if not self.train:
            m = self.matches[index]
            img1 = transform_img(self.data[m[0]])
            img2 = transform_img(self.data[m[1]])
            return img1, img2, m[2]

        t = self.triplets[index]
        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]

        img_a = transform_img(a)
        img_p = transform_img(p)
        img_n = None
        if self.out_triplets:
            img_n = transform_img(n)
        # transform images if required
        if self.fliprot:
            do_flip = random.random() > 0.5
            do_rot = random.random() > 0.5
            if do_rot:
                img_a = img_a.permute(0,2,1)
                img_p = img_p.permute(0,2,1)
                if self.out_triplets:
                    img_n = img_n.permute(0,2,1)
            if do_flip:
                img_a = torch.from_numpy(deepcopy(img_a.numpy()[:,:,::-1]))
                img_p = torch.from_numpy(deepcopy(img_p.numpy()[:,:,::-1]))
                if self.out_triplets:
                    img_n = torch.from_numpy(deepcopy(img_n.numpy()[:,:,::-1]))
        if self.out_triplets:
            return (img_a, img_p, img_n)
        else:
            return (img_a, img_p)
    
    def generate_triplets(self, labels, num_triplets):
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

#         for x in tqdm(range(num_triplets)):
#             if len(already_idxs) >= self.batch_size:
#                 already_idxs = set()
#             c1 = np.random.randint(0, n_classes)
#             while c1 in already_idxs:
#                 c1 = np.random.randint(0, n_classes)
#             already_idxs.add(c1)
#             c2 = np.random.randint(0, n_classes)
#             while c1 == c2:
#                 c2 = np.random.randint(0, n_classes)
#             if len(indices[c1]) == 2:  # hack to speed up process
#                 n1, n2 = 0, 1
#             else:
#                 n1 = np.random.randint(0, len(indices[c1]))
#                 n2 = np.random.randint(0, len(indices[c1]))
#                 while n1 == n2:
#                     n2 = np.random.randint(0, len(indices[c1]))
#             n3 = np.random.randint(0, len(indices[c2]))
#             triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
#         return torch.LongTensor(np.array(triplets))

        for x in tqdm(range(num_triplets)):
            x = 0
            if len(already_idxs) >= self.batch_size:
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
            triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])

        return torch.LongTensor(np.array(triplets))


    def __len__(self):
        if self.train:
            return self.triplets.size(0)
        else:
            return self.matches.size(0)

    def _check_datafile_exists(self):
        return os.path.exists(self.data_file)
