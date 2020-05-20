import torch

from models.superpoint import SuperPoint
from models.superglue import SuperGlue


class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """
    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config['superpoint'])
        self.superglue = SuperGlue(config['superglue'])

    def forward(self, data):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}

        # Extract keypoints, scores and descriptors using SuperPoint model
        pred0 = self.superpoint({'image': data['image0']})
        pred = {**pred, **{k+'0': v for k, v in pred0.items()}}

        pred1 = self.superpoint({'image': data['image1']})
        pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

        # return pred
        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}
        for k, v in data.items():
            print(k, type(v))
        print()
        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])
        for k, v in data.items():
            print(k, v.shape)

        # return self.superglue(data)
        # Perform the matching
        pred = {**pred, **self.superglue(data)}

        return pred
