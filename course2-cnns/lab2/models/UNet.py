import torch.nn as nn
import torch


class UNet(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(UNet, self).__init__()
        # ------------------------ Laboratoire 2 - Question 4 - Début de la section à compléter ------------------------
        self.hidden = 32

        # Down 1
        self.conv_1_1 = nn.Conv2d(input_channels, 32, (3,3), (1,1), (1,1))
        self.relu_1_1 = nn.ReLU()
        self.conv_1_2 = nn.Conv2d(32, 32, (3,3), (1,1), (1,1))
        self.relu_1_2 = nn.ReLU()

        # Down 2
        self.maxpool_2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv_2_1 = nn.Conv2d(32, 64, (3,3), (1,1), (1,1))
        self.relu_2_1 = nn.ReLU()
        self.conv_2_2 = nn.Conv2d(64, 64, (3,3), (1,1), (1,1))
        self.relu_2_2 = nn.ReLU()

        # Down 3
        self.maxpool_3 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv_3_1 = nn.Conv2d(64, 128, (3,3), (1,1), (1,1))
        self.relu_3_1 = nn.ReLU()
        self.conv_3_2 = nn.Conv2d(128, 128, (3,3), (1,1), (1,1))
        self.relu_3_2 = nn.ReLU()

        # Down 4
        self.maxpool_4 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv_4_1 = nn.Conv2d(128, 256, (3,3), (1,1), (1,1))
        self.relu_4_1 = nn.ReLU()
        self.conv_4_2 = nn.Conv2d(256, 256, (3,3), (1,1), (1,1))
        self.relu_4_2 = nn.ReLU()

        # Down 5
        self.maxpool_5 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv_5_1 = nn.Conv2d(256, 512, (3,3), (1,1), (1,1))
        self.relu_5_1 = nn.ReLU()
        self.conv_5_2 = nn.Conv2d(512, 256, (3,3), (1,1), (1,1))
        self.relu_5_2 = nn.ReLU()

        # Up 6
        self.upsample_6 = nn.ConvTranspose2d(256, 256, (2,2), (2,2), (0,0))
        self.conv_6_1 = nn.Conv2d(512, 256, (3,3), (1,1), (1,1))
        self.relu_6_1 = nn.ReLU()
        self.conv_6_2 = nn.Conv2d(256, 128, (3,3), (1,1), (1,1))
        self.relu_6_2 = nn.ReLU()

        # Up 7
        self.upsample_7 = nn.ConvTranspose2d(128, 128, (2,2), (2,2), (0,0))
        self.conv_7_1 = nn.Conv2d(256, 128, (3,3), (1,1), (1,1))
        self.relu_7_1 = nn.ReLU()
        self.conv_7_2 = nn.Conv2d(128, 64, (3,3), (1,1), (1,1))
        self.relu_7_2 = nn.ReLU()

        # Up 8
        self.upsample_8 = nn.ConvTranspose2d(64, 64, (2,2), (2,2), (0,0))
        self.conv_8_1 = nn.Conv2d(128, 64, (3,3), (1,1), (1,1))
        self.relu_8_1 = nn.ReLU()
        self.conv_8_2 = nn.Conv2d(64, 32, (3,3), (1,1), (1,1))
        self.relu_8_2 = nn.ReLU()

        # Up 9
        self.upsample_9 = nn.ConvTranspose2d(32, 32, (2,2), (2,2), (0,0))
        self.conv_9_1 = nn.Conv2d(64, 32, (3,3), (1,1), (1,1))
        self.relu_9_1 = nn.ReLU()
        self.conv_9_2 = nn.Conv2d(32, 32, (3,3), (1,1), (1,1))
        self.relu_9_2 = nn.ReLU()

        self.output_conv = nn.Conv2d(self.hidden, n_classes, kernel_size=1)

    def forward(self, x):
        # Down 1
        x1 = self.conv_1_1(x)
        x1 = self.relu_1_1(x1)
        x1 = self.conv_1_2(x1) 
        x1 = self.relu_1_2(x1) 

        # Down 2
        x2 = self.maxpool_2(x1) 
        x2 = self.conv_2_1(x2)
        x2 = self.relu_2_1(x2)
        x2 = self.conv_2_2(x2) 
        x2 = self.relu_2_2(x2) 

        # Down 3
        x3 = self.maxpool_3(x2) 
        x3 = self.conv_3_1(x3)
        x3 = self.relu_3_1(x3) 
        x3 = self.conv_3_2(x3)
        x3 = self.relu_3_2(x3)

        # Down 4
        x4 = self.maxpool_4(x3) 
        x4 = self.conv_4_1(x4)
        x4 = self.relu_4_1(x4)
        x4 = self.conv_4_2(x4)
        x4 = self.relu_4_2(x4)

        # Down 5
        x5 = self.maxpool_5(x4)
        x5 = self.conv_5_1(x5)
        x5 = self.relu_5_1(x5)
        x5 = self.conv_5_2(x5)
        x5 = self.relu_5_2(x5)

        # Up 6
        x6 = self.upsample_6(x5)
        x6 = torch.cat((x6, x4), dim=1)
        x6 = self.conv_6_1(x6)
        x6 = self.relu_6_1(x6)
        x6 = self.conv_6_2(x6)
        x6 = self.relu_6_2(x6)

        # Up 7
        x7 = self.upsample_7(x6) 
        x7 = torch.cat((x7, x3), dim=1)
        x7 = self.conv_7_1(x7)
        x7 = self.relu_7_1(x7)
        x7 = self.conv_7_2(x7)
        x7 = self.relu_7_2(x7)

        # Up 8
        x8 = self.upsample_8(x7)
        x8 = torch.cat((x8, x2), dim=1)
        x8 = self.conv_8_1(x8)
        x8 = self.relu_8_1(x8)
        x8 = self.conv_8_2(x8)
        x8 = self.relu_8_2(x8)

        # Up 9
        x9 = self.upsample_9(x8)
        x9 = torch.cat((x9, x1), dim=1)
        x9 = self.conv_9_1(x9)
        x9 = self.relu_9_1(x9)
        x9 = self.conv_9_2(x9)
        x9 = self.relu_9_2(x9)

        # Out
        out = self.output_conv(x9)

        return out
        # ------------------------ Laboratoire 2 - Question 4 - Fin de la section à compléter --------------------------
