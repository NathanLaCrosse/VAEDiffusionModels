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

        self.norm1 = nn.GroupNorm(bottleneck_channels//4, bottleneck_channels)
        self.norm2 = nn.GroupNorm(initial_channels//4, initial_channels)

    def forward(self, x):
        bottleneck = self.norm1(self.conv1(x))
        bottleneck = F.silu(self.conv2(bottleneck))
        bottleneck = self.norm2(F.silu(self.conv3(bottleneck)))

        if self.dropout_p > 0:
            bottleneck = F.dropout(bottleneck, p=self.dropout_p)

        x = F.silu(x + bottleneck)
        return x

# class ResidualBlockWithEmbeddings(nn.Module):
#     def __init__(self, initial_channels, bottleneck_channels, im_dim, num_labels, dropout_p=0.0):
#         super(ResidualBlockWithEmbeddings, self).__init__()
#
#         self.dropout_p = dropout_p
#
#         self.label_embedding = nn.Embedding(num_labels, im_dim)
#         self.embedding_mlp = nn.Sequential(
#             nn.Linear(im_dim, im_dim*4)
#         )
#
#         self.conv1 = nn.Conv2d(initial_channels, bottleneck_channels, 1)
#         self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, 1, 1)
#         self.conv3 = nn.Conv2d(initial_channels, bottleneck_channels, 1)

class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()

