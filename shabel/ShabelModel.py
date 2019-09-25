""" This module contains CNN models """

import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    """Convolutional neural network - 2 conv layer, 3 fully connected"""

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(96800, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = x.view(-1, self.num_flat_features(x)) # to vector form
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x) -> int:
        """ Compute the number of remaining image pixels for one sample.
            Note: Code was taken out from a pytorch example
        """
        size = x.size()[1:]  # compute the number of remaining image pixels for one sample
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
