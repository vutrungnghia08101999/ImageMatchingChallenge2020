from torchvision.models import resnet18
import torch.nn as nn
import torch


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class MyModel(nn.Module):
    def __init__(self, num_features=128):
        super(MyModel, self).__init__()
        
        self.backbone = resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_features)
        self.l2norm = L2Norm()
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.l2norm(x)
        return x
    
model = MyModel()
x = torch.randn(size=(2,3,64,64))
print(model(x).shape)