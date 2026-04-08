"""
ResNet34 Multi-Scale Encoder.

256×256 입력 기준:
    layer1: [B,  64, 64, 64]
    layer2: [B, 128, 32, 32]
    layer3: [B, 256, 16, 16]
    layer4: [B, 512,  8,  8]
"""

import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet34(pretrained=pretrained)

        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1,
                                    resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.channels = [64, 128, 256, 512]

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]
