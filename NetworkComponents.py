import math
import torch
from pandas.core.nanops import bottleneck_switch
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

class ResidualBlockWithEmbeddings(nn.Module):
    def __init__(self, initial_channels, bottleneck_channels, im_dim, time_embed_dim=64, label_embed_dim=256, dropout_p=0.0):
        """
        Convolutional Neural Network block that utilizes a residual connection to retain gradients.
        Incorporates a time and label embedding for use in a diffusion model.

        :param initial_channels: The initial channels in the input image (output also has initial_channels)
        :param bottleneck_channels: The intermediate channel count when the 3x3 convolution is applied.
        :param im_dim: Side length of the image (assumed to be square)
        :param time_embed_dim: Dimension of the time embedding
        :param label_embed_dim: Dimension of the label embedding
        :param dropout_p: Dropout probability
        """
        super(ResidualBlockWithEmbeddings, self).__init__()

        self.initial_channels = initial_channels
        self.bottleneck_channels = bottleneck_channels
        self.dropout_p = dropout_p
        self.im_dim = im_dim

        # Parse data out of the global time information into data that's only needed
        # on the local scale
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim*4),
            nn.SiLU(),
            # nn.Dropout(dropout_p),
            nn.Linear(time_embed_dim*4, bottleneck_channels),
            # nn.SiLU()
        )

        # Similar to the time mlp, this draws out locally important information of out the
        # global label information
        # self.label_mlp = nn.Sequential(
        #     nn.Linear(label_embed_dim, label_embed_dim),
        #     nn.SiLU(),
        #     # nn.Dropout(dropout_p),
        #     nn.Linear(label_embed_dim,im_dim**2),
        #     # nn.SiLU()
        # )

        self.conv1 = nn.Conv2d(initial_channels, bottleneck_channels, 1)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(bottleneck_channels, initial_channels, 1)

        # Extra normalization stuff
        self.norm1 = nn.GroupNorm(8, bottleneck_channels)
        self.norm2 = nn.GroupNorm(8, initial_channels)

    def forward(self, x, t_vect):
        """
        Forward pass of the residual block.
        :param x: An input image of size (B x C x H x W)
        :param t_vect: A time embedding of size (B X E) [E is determined by constructor]
        :param l_vect: A label embedding of size (B X L) [L is determined by constructor]
        :return: x - Output of this neural network block of size (B x C x H x W)
        """
        batch_size, c, rows, cols = x.size()

        # Create local context encodings of t and l
        local_t = self.time_mlp(t_vect)
        # local_l = self.label_mlp(l_vect)

        # First convolution
        res = F.silu(self.norm1(self.conv1(x)))

        # local_t has length bottleneck channels -> convert into a view so that
        # it can be added to res
        res = res + local_t[:, :, None, None]

        # local_l has length dim**2 -> convert into a different view added to resp
        # local_l = local_l.view(batch_size, self.im_dim, self.im_dim)
        # res = res + local_l[:, None, :, :]
        # res = res + local_l[:, :, None, None]

        # Perform the rest of the convolutions
        res = F.silu(res)
        res = self.conv2(res)
        res = self.norm2(F.silu(self.conv3(res)))

        if self.dropout_p > 0:
            res = F.dropout(res, p=self.dropout_p)

        # Residual connection
        return x + res / math.sqrt(2)

class NResBlocks(nn.Module):

    def __init__(self, n, initial_channels, bottleneck_channels, im_dim, time_embed_dim=64, label_embed_dim=256, dropout_p=0.0):
        super(NResBlocks, self).__init__()

        self.blocks = nn.ModuleList(
            [ResidualBlockWithEmbeddings(initial_channels, bottleneck_channels, im_dim, time_embed_dim, label_embed_dim, dropout_p) for i in range(n)]
        )

    def forward(self, x, t_vect):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, t_vect)
        return x

def positional_encoding(seq_len, dim):
    """
    Generates a positional encoding matrix of shape (seq_len, dim) using
    sinusoidal functions as described in the original Transformer paper.
    """
    pos = torch.arange(seq_len).unsqueeze(1)  # shape: (seq_len, 1)
    i = torch.arange(dim).unsqueeze(0)  # shape: (1, dim)
    omega = 1 / torch.pow(10000, (2 * (i // 2)) / dim)  # frequency term
    angles = pos * omega  # outer product: position * frequency

    pos_enc = torch.zeros(seq_len, dim)
    pos_enc[:, 0::2] = torch.sin(angles[:, 0::2])  # apply sin to even indices
    pos_enc[:, 1::2] = torch.cos(angles[:, 1::2])  # apply cos to odd indices
    return pos_enc

if __name__ == '__main__':
    resblock = ResidualBlockWithEmbeddings(16, 7, 2, 64, 256)

    test = torch.randn((3, 16, 2, 2))
    t_vect = torch.randn((3,64))
    l_vect = torch.randn((3,256))
    out = resblock(test, t_vect, l_vect)

    print('hi')

