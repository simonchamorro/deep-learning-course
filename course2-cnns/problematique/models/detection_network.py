import torch.nn as nn


class DetectionNetwork(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(DetectionNetwork, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(input_channels, 4, (3, 3), (1, 1), (1, 1))
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, (3,3), (1,1), (1,1))
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, (3,3), (1,1), (1,1))
        self.bn3 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.fc1 = nn.Linear(16*13*13, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, n_classes*5)


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

        x = self.fc1(x.view(x.shape[0], -1))
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        output = self.fc3(x)
        return output.view(output.shape[0], 3,5)
