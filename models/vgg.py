import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class VGGSNN(nn.Module):
    def __init__(self, tau=0.5):
        super(VGGSNN, self).__init__()
        global BN
        BN = nn.BatchNorm2d
        global ReLU
        ReLU = nn.ReLU
        pool = nn.AvgPool2d
        # pool = APLayer(2)
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            ReLU(),
            pool(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            ReLU(),
            pool(2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            ReLU(),
            pool(2),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            ReLU(),
            pool(2),
        )
        W = int(48 / 2 / 2 / 2 / 2)
        # self.T = 10
        self.classifier = nn.Sequential(
        # nn.Dropout(0.25),
        nn.Linear(512 * W * W, 10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # print(input.shape)
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x