import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # ---------------------- Laboratoire 1 - Question 5 et 6 - Début de la section à compléter ---------------------
        # # Linear model
        # self.fc1 = nn.Linear(28 * 28, 10)

        # # Question 5
        # self.conv1 = nn.Conv2d(1, 1, (3,3), (1,1), (1,1))
        # self.fc1 = nn.Linear(14 * 14, 10)
        # self.pool = nn.MaxPool2d(kernel_size=(2,2))
        # self.relu = nn.ReLU()

        # Question 6
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 4, (3,3), (1,1), (1,1))
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 2, (3,3), (1,1), (1,1))
        self.bn2 = nn.BatchNorm2d(2)
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.fc1 = nn.Linear(2*7*7, 10)
        # ---------------------- Laboratoire 1 - Question 5 et 6 - Fin de la section à compléter -----------------------

    def forward(self, x):
        # ---------------------- Laboratoire 1 - Question 5 et 6 - Début de la section à compléter ---------------------
        # # Linear model
        # output = self.fc1(x.view(x.shape[0], -1))

        # # Question 5
        # x = self.pool(self.relu(self.conv1(x)))
        # output = self.fc1(x.view(x.shape[0], -1))

        # Question 6
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        output = self.fc1(x.view(x.shape[0], -1))
        # ---------------------- Laboratoire 1 - Question 5 et 6 - Fin de la section à compléter -----------------------
        return output
