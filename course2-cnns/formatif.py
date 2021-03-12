#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Q5Network(nn.Module):
    def __init__(self):
        super(Q5Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv5 = nn.Conv2d(32, 10, 8, 1, 0)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.softmax(x)
        return x


test = Q5Network()
input_tensor = torch.zeros((256, 3, 128, 128))
output = test(input_tensor)
import pdb; pdb.set_trace()