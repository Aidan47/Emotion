import torch
import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        
        # 1st conv layer
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # 2nd conv layer
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # 3rd conv layer
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        # dense layers to output
        self.d1 = nn.Linear(9216, 256)
        self.d2 = nn.Linear(256, 7)
        
    def forward(self, x):
        x = nn.MaxPool2d(2)(F.relu((self.bn1(self.conv1(x)))))
        x = nn.MaxPool2d(2)(F.relu((self.bn2(self.conv2(x)))))
        x = nn.MaxPool2d(2)(F.relu((self.bn3(self.conv3(x)))))
        x = torch.flatten(x, 1)
        x = F.relu(self.d1(x))
        return self.d2(x)