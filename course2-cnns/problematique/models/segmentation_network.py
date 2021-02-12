import torch.nn as nn
import torch



class SegmentationNetwork(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(SegmentationNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2,2))

        # Convolution
        self.conv1 = nn.Conv2d(in_channels, 8, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(8, 32, (3,3), (1,1), (1,1))
        self.conv3 = nn.Conv2d(32, 64, (3,3), (1,1), (1,1))
        self.conv4 = nn.Conv2d(64, 128, (3,3), (1,1), (1,1))
        
        # De convolution
        self.convt1 = nn.ConvTranspose2d(128, 64, (3,3), (2,2), (0,0))
        self.convt2 = nn.ConvTranspose2d(64, 32, (2,2), (2, 2), (0,0))
        self.convt3 = nn.ConvTranspose2d(32, 8, (3,3), (2,2), (0,0))
        self.conv5 = nn.Conv2d(128, 64, (3,3), (1,1), (1,1))
        self.conv6 = nn.Conv2d(64, 32, (3,3), (1,1), (1,1))
        self.conv7 = nn.Conv2d(16, 8, (3,3), (1,1), (1,1))
        self.conv8 = nn.Conv2d(8, n_classes + 1, (3,3), (1,1), (1,1))

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu(x1)

        x2 = self.pool(x1)
        x2 = self.conv2(x2)
        x2 = self.relu(x2)

        x3 = self.pool(x2)
        x3 = self.conv3(x3)
        x3 = self.relu(x3)

        x4 = self.pool(x3)
        x4 = self.conv4(x4)
        x4 = self.relu(x4)

        x5 = self.convt1(x4)
        x5 = torch.cat((x5, x3), dim=1)
        x5 = self.conv5(x5)
        x5 = self.relu(x5)

        x6 = self.convt2(x5)
        x6 = torch.cat((x6, x2), dim=1)
        x6 = self.conv6(x6)
        x6 = self.relu(x6)

        x7 = self.convt3(x6)
        x7 = torch.cat((x7, x1), dim=1)
        x7 = self.conv7(x7)
        x7 = self.relu(x7)
        output = self.conv8(x7)
        return output
