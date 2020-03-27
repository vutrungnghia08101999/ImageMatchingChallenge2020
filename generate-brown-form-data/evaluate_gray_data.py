import logging
from tqdm import tqdm

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import TripletPhotoTour
from models import HardNet

logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

torch.manual_seed(0)

############### PARAMETERS ##############
DATAROOT = '/media/vutrungnghia/New Volume/P2-ImageMatchingChallenge/dataset/brown-dataset'
DATANAME = 'yosemite'
#########################################

logging.info('=========================================================')
logging.info('\nEVALUATION DATA\n')
logging.info(f'dataroot: {DATAROOT}')
logging.info(f'dataname: {DATANAME}')

cv2_scale36 = lambda x: cv2.resize(x, dsize=(36, 36),
                                 interpolation=cv2.INTER_LINEAR)
cv2_scale = lambda x: cv2.resize(x, dsize=(32, 32),
                                 interpolation=cv2.INTER_LINEAR)

np_reshape = lambda x: np.reshape(x, (32, 32, 1))

        
def create_dataset(name: str, is_train: bool, load_random_triplet: bool):
    transform = transforms.Compose([
        transforms.Lambda(cv2_scale),
        transforms.Lambda(np_reshape),
        transforms.ToTensor(),
        transforms.Normalize((0.443728476019,), (0.20197947209,))])
    
    dataset = TripletPhotoTour(n_triplets=5000,
                               train=is_train,
                               load_random_triplets = load_random_triplet,
                               batch_size=1024,
                               root=DATAROOT,
                               name=name,
                               transform=transform)
    return dataset


def ErrorRateAt95Recall(labels, scores):
    distances = 1.0 / (scores + 1e-8)
    recall_point = 0.95
    labels = labels[np.argsort(distances)]
    # Sliding threshold: get first index where recall >= recall_point. 
    # This is the index where the number of elements with label==1 below the threshold reaches a fraction of 
    # 'recall_point' of the total number of elements with label==1. 
    # (np.argmax returns the first occurrence of a '1' in a bool array). 
    threshold_index = np.argmax(np.cumsum(labels) >= recall_point * np.sum(labels)) 

    FP = np.sum(labels[:threshold_index] == 0) # Below threshold (i.e., labelled positive), but should be negative
    TN = np.sum(labels[threshold_index:] == 0) # Above threshold (i.e., labelled negative), and should be negative
    return float(FP) / float(FP + TN)


def test(test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:

        with torch.no_grad():
            out_a = model(data_a)
            out_p = model(data_p)
        dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy().reshape(-1,1))
        ll = label.data.cpu().numpy().reshape(-1, 1)
        labels.append(ll)

        if batch_idx % 10 == 0:
            pbar.set_description(' Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset),
                       100. * batch_idx / len(test_loader)))

    num_tests = test_loader.dataset.matches.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    distances = np.vstack(distances).reshape(num_tests)

    fpr95 = ErrorRateAt95Recall(labels, 1.0 / (distances + 1e-8))
    logging.info('\33[91mTest set: Accuracy(FPR95): {:.8f}\n\33[0m'.format(fpr95))

    return


test_dataloader = DataLoader(
    create_dataset(DATANAME, False, False),
    batch_size=1024,
    shuffle=False,
    num_workers=0
)

model = HardNet()
checkpoint = torch.load('gray_hardnet_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
logging.info(f'Test on {DATANAME} dataset')
test(test_dataloader, model, 0)

dataset = create_dataset(DATANAME, True, True)
pos_distances = []
neg_distances = []
model.eval()
for triplet in tqdm(dataset):
    triplet = dataset[0]
    x = triplet[0].view(1, 1, 32, 32)
    y = triplet[1].view(1, 1, 32, 32)
    z = triplet[2].view(1, 1, 32, 32)
    with torch.no_grad():
        x = model(x).squeeze()
        y = model(y).squeeze()
        z = model(z).squeeze()
    pos_distances.append(float(torch.norm(x-y)))
    neg_distances.append(float(torch.norm(x-z)))


ratio = np.array(neg_distances).mean()/np.array(pos_distances).mean()
logging.info(f'average neg_distance/pos_distance: {ratio} on {len(dataset)} triplets')
logging.info('Evaluation completed\n')
