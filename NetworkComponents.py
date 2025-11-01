import torch
from torch import nn
import torch.nn.functional as F

class ConvolutionalBlock(nn.Module):
    """
    Performs two convolution + relu blocks.
    """
    def __init__(self, c1, c2, c3, rows, cols, dropout_p=0.0):
        super(ConvolutionalBlock, self).__init__()

        self.dropout_p = dropout_p
        self.conv1 = nn.Conv2d(c1, c2, 3, 1, 1)
        self.conv2 = nn.Conv2d(c2, c3, 3, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        if self.dropout_p > 0:
            x = F.dropout(x, p=self.dropout_p)

        return x

class BottleneckBlock(nn.Module):
    """
    Implements a bottleneck block similar to the ones utilized by ResNets
    """
    def __init__(self, initial_channels, bottleneck_channels, dropout_p=0.0):
        super(BottleneckBlock, self).__init__()

        self.dropout_p = dropout_p
        self.conv1 = nn.Conv2d(initial_channels, bottleneck_channels, 1)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(bottleneck_channels, initial_channels, 1)

    def forward(self, x):
        bottleneck = self.conv1(x)
        bottleneck = F.relu(self.conv2(bottleneck))
        bottleneck = F.relu(self.conv3(bottleneck))

        if self.dropout_p > 0:
            bottleneck = F.dropout(bottleneck, p=self.dropout_p)

        x = F.relu(x + bottleneck)
        return x
