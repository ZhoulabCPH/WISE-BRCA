import torch
import torchvision.models as models
import torch.nn.functional as F

from torch import nn

class Tumour_segmentation(nn.Module):
    def __init__(self, ):
        super(Tumour_segmentation, self).__init__()

        self.resnet = models.resnet50(pretrained=True)

        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 2)
        )
        for layer in self.resnet.fc:
            if type(layer) == torch.nn.modules.linear.Linear:
                nn.init.xavier_normal_(layer.weight)


    def forward(self, x):
        y = self.resnet(x)

        return y
