import torch.nn as nn
import torch



class SegmentationNetwork(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(SegmentationNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2,2))

        # Convolution
        self.conv1 = nn.Conv2d(in_channels, 16, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(16, 32, (3,3), (1,1), (1,1))
        self.conv4 = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(32, 64, (3,3), (1,1), (1,1))
        self.conv6 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        
        # De convolution
        self.convt1 = nn.ConvTranspose2d(32, 32, (2, 2), (2,2), (0,0))
        self.conv7 = nn.Conv2d(64, 32, (3,3), (1,1), (1,1))
        self.conv8 = nn.Conv2d(32, 16, (3,3), (1,1), (1,1))
        
        self.convt2 = nn.ConvTranspose2d(16, 16, (3,3), (2,2), (0,0))
        self.conv9 = nn.Conv2d(32, 16, (3,3), (1,1), (1,1))
        self.conv10 = nn.Conv2d(16, 16, (3,3), (1,1), (1,1))
        self.conv11 = nn.Conv2d(16, n_classes + 1, (3,3), (1,1), (1,1))

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.relu(x1)

        x2 = self.pool(x1)
        x2 = self.conv3(x2)
        x2 = self.relu(x2)
        x2 = self.conv4(x2)
        x2 = self.relu(x2)

        x3 = self.pool(x2)
        x3 = self.conv5(x3)
        x3 = self.relu(x3)
        x3 = self.conv6(x3)
        x3 = self.relu(x3)

        x4 = self.convt1(x3)
        x4 = torch.cat((x4, x2), dim=1)
        x4 = self.conv7(x4)
        x4 = self.relu(x4)
        x4 = self.conv8(x4)
        x4 = self.relu(x4)

        x5 = self.convt2(x4)
        x5 = torch.cat((x5, x1), dim=1)
        x5 = self.conv9(x5)
        x5 = self.relu(x5)
        x5 = self.conv10(x5)
        x5 = self.relu(x5)
        output = self.conv11(x5)
        return output
