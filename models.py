import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn as nn
from utils import L2Norm


class HardNet(nn.Module):
    """HardNet model definition
    """
    def __init__(self):
        super(HardNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias = False),
            nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

    
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, gain=0.6)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return


class ResNet(nn.Module):
    def __init__(self, num_features=128):
        super(ResNet, self).__init__()
        
        self.backbone = resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_features)
        self.l2norm = L2Norm()
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.l2norm(x)
        return x


eps_fea_norm = 1e-5
eps_l2_norm = 1e-10

class SOSNet32x32(nn.Module):
    """
    128-dimensional SOSNet model definition trained on 32x32 patches
    """
    def __init__(self, dim_desc=128, drop_rate=0.1):
        super(SOSNet32x32, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate

        norm_layer = nn.BatchNorm2d
        activation = nn.ReLU()

        self.layers = nn.Sequential(
            nn.InstanceNorm2d(3, affine=False, eps=eps_fea_norm),
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32, affine=False, eps=eps_fea_norm),
            activation,

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(64, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64, affine=False, eps=eps_fea_norm),
            activation,

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
            activation,

            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
        )

        self.desc_norm = nn.Sequential(
            nn.LocalResponseNorm(2 * self.dim_desc, alpha=2 * self.dim_desc, beta=0.5, k=0)
        )

        return

    def forward(self, patch):
        descr = self.desc_norm(self.layers(patch) + eps_l2_norm)
        descr = descr.view(descr.size(0), -1)
        return descr
