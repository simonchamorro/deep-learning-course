import torch.nn as nn
import torch


class DetectionNetwork(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(DetectionNetwork, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.sigmoid  = nn.Sigmoid()
        self.conv1 = nn.Conv2d(input_channels, 32, (5,5), (1, 1), (1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (3,3), (1,1), (1,1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, (3,3), (1,1), (1,1))
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.fc1 = nn.Linear(32*6*6, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, n_classes*7)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.fc1(x.view(x.shape[0], -1))
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        output = x.view(x.shape[0], 3, 7)
        return output
