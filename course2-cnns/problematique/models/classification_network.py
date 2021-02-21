import torch.nn as nn


class ClassificationNetwork(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(ClassificationNetwork, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, 16, (5,5), (1, 1), (1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, (3,3), (1,1), (1,1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (3,3), (1,1), (1,1))
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
        self.fc1 = nn.Linear(5*5*64, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, n_classes)


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
        output = self.fc3(x)
        return output
