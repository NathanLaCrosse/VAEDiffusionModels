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
            nn.Dropout(dropout_p),
            nn.Linear(time_embed_dim*4, bottleneck_channels),
            # nn.SiLU()
        )

        # Similar to the time mlp, this draws out locally important information of out the
        # global label information
        # self.label_mlp = nn.Sequential(
        #     nn.Linear(label_embed_dim, label_embed_dim),
        #     nn.SiLU(),
        #     nn.Dropout(dropout_p),
        #     nn.Linear(label_embed_dim,im_dim**2)
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
        # local_l = self.label_mlp(l_vect).view(batch_size, 1, self.im_dim, self.im_dim).contiguous()

        # First convolution
        res = self.norm1(F.silu(self.conv1(x)))

        # local_t has length bottleneck channels -> convert into a view so that
        # it can be added to res. Then apply swish to zero out unimportant information
        res = self.conv2(res)
        res = res + local_t[:, :, None, None]
        res = F.silu(res)

        # Concatenate with label embedding to gain conditionality
        # res = torch.cat([res, local_l], dim=1)

        # Perform the rest of the convolutions
        # res = self.conv2(res)
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



class MultiHeadSelfAttention(nn.Module):

    def __init__(self, input_channels, heads=4, dropout=0.0):
        super(MultiHeadSelfAttention, self).__init__()

        self.atten = MultiHeadedAttention(input_channels, heads, dropout=dropout)

    def forward(self, x):
        # X Shape: (B, C, H, W)
        batches, channels, rows, cols = x.size()

        # Reshape X to be of size (B, H*W, C)
        x = x.view(batches, channels, rows * cols) # -> (B, C, H*W)
        x = x.permute(0, 2, 1).contiguous() # -> (B, H*W, C)

        # Apply the attention mechanism
        x = self.atten(x)

        # Reshape X back to size (B, C, H, W)
        x = x.permute(0, 2, 1) # -> (B, C, H*W)
        x = x.view(batches, channels, rows, cols).contiguous() # -> (B, C, H, W)

        return x



class CrossAttention(nn.Module):
    def __init__(self, channel_count, label_embed_size, num_heads, dropout):
        super(CrossAttention, self).__init__()
        assert channel_count % num_heads == 0, \
            "input_size must be divisible by num_heads for even head splitting."

        self.num_heads = num_heads
        self.head_dim = channel_count // num_heads

        # Projection to map label embedding to a tensor
        self.label_projection = nn.Linear(label_embed_size, channel_count)

        # Linear projections for Query, Key, and Value
        self.query_projection = nn.Linear(channel_count, channel_count)
        self.key_projection = nn.Linear(channel_count, channel_count)
        self.value_projection = nn.Linear(channel_count, channel_count)

        # Output projection after attention process
        self.output_projection = nn.Linear(channel_count, channel_count)

        # Feedforward for added nonlinearity
        self.feedforward = nn.Sequential(
            nn.Linear(channel_count, channel_count*4),
            nn.SiLU(),
            nn.Linear(channel_count*4, channel_count)
        )

        # Regularization
        self.layer_norm1 = nn.LayerNorm(channel_count)
        self.layer_norm2 = nn.LayerNorm(channel_count)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, label_emb):
        # Inputs
        # X: (B, C, H, W)
        # L: label - stored as long
        batches, channels, rows, cols = x.size()

        # Reshape X: (B, C, H, W) -> (B, H*W, C)
        x = x.view(batches, channels, rows*cols).permute(0, 2, 1).contiguous()
        normed = self.layer_norm1(x)

        # Use label to generate keys and values
        label_emb = label_emb.unsqueeze(1) # (B, 1, label_embed_size)
        embedded = self.label_projection(label_emb) # (B, 1, channel_count)
        K = self.key_projection(embedded).reshape(batches, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_projection(embedded).reshape(batches, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Use X to generate queries
        Q = self.query_projection(normed).view(batches, rows*cols, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = attention_weights @ V  # (B, H, S, d_h)

        # Combine attention heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batches, rows*cols, self.num_heads*self.head_dim)
        attention_output = self.output_projection(attention_output)

        x = x + self.dropout(attention_output)
        x = x + self.dropout(self.feedforward(self.layer_norm2(x)))

        # Reshape X back to size (B, C, H, W)
        x = x.permute(0, 2, 1)  # -> (B, C, H*W)
        x = x.view(batches, channels, rows, cols).contiguous()  # -> (B, C, H, W)

        return x




class MultiHeadedAttention(nn.Module):
    """
    Implements one layer of a Transformer encoder, consisting of:
    1. Multi-head self-attention sublayer (with residual connection + LayerNorm)
    2. Feedforward sublayer (with residual connection + LayerNorm)
    """

    def __init__(self, input_size, num_heads, dropout):
        super().__init__()
        assert input_size % num_heads == 0, \
            "input_size must be divisible by num_heads for even head splitting."

        self.num_heads = num_heads
        self.head_dim = input_size // num_heads

        # Linear projections for Query, Key, and Value
        self.query_projection = nn.Linear(input_size, input_size)
        self.key_projection = nn.Linear(input_size, input_size)
        self.value_projection = nn.Linear(input_size, input_size)

        # Output projection after concatenating all heads
        self.output_projection = nn.Linear(input_size, input_size)

        # Feedforward for added nonlinearity
        self.feedforward = nn.Sequential(
            nn.Linear(input_size, input_size*4),
            nn.SiLU(),
            nn.Linear(input_size*4, input_size)
        )

        # Norm and dropout for regularization
        self.layer_norm1 = nn.LayerNorm(input_size)
        self.layer_norm2 = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_embeddings):
        """
        Args:
            input_embeddings (Tensor): shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, sequence_length, embedding_dim = input_embeddings.shape
        num_heads = self.num_heads
        head_dim = self.head_dim

        h = self.layer_norm1(input_embeddings) # Pre-computation norm -> more stable gradients

        # ---------------------------
        # Multi-Head Self-Attention
        # ---------------------------
        Q = self.query_projection(h).view(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)
        K = self.key_projection(h).view(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)
        V = self.value_projection(h).view(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = (Q @ K.transpose(-2, -1)) / (head_dim ** 0.5)  # (B, H, S, S)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = attention_weights @ V  # (B, H, S, d_h)

        # Combine attention heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dim)
        attention_output = self.output_projection(attention_output)

        # Add & do prenorm on x
        x = input_embeddings + self.dropout(attention_output)
        x = x + self.dropout(self.feedforward(self.layer_norm2(x)))

        return x  # (B, S, D)

class UNetLayer(nn.Module):

    def __init__(self, channels, im_dim, time_embed_dim, label_embed_dim, dropout_p=0.0):
        super(UNetLayer, self).__init__()

        self.res_block1 = ResidualBlockWithEmbeddings(channels, channels//2, im_dim, time_embed_dim, label_embed_dim, dropout_p)
        self.res_block2 = ResidualBlockWithEmbeddings(channels, channels//2, im_dim, time_embed_dim, label_embed_dim, dropout_p)
        self.cross = CrossAttention(channels, label_embed_dim, channels//16, dropout_p)
        self.self_atten = MultiHeadSelfAttention(channels, channels//16, dropout_p)

    def forward(self, x, t_embed, label_embed):
        x = self.res_block1(x, t_embed)
        x = self.res_block2(x, t_embed)
        x = self.cross(x, label_embed)
        return self.self_atten(x)


if __name__ == '__main__':
    # resblock = ResidualBlockWithEmbeddings(16, 7, 2, 64, 256)

    # test = torch.randn((3, 16, 2, 2))
    # t_vect = torch.randn((3,64))
    # l_vect = torch.randn((3,256))
    # out = resblock(test, t_vect, l_vect)

    # atten = MultiHeadedAttention(8, 2, 0.0)
    # test = torch.randn((2, 3, 8))
    # out = atten(test)

    # atten = MultiHeadSelfAttention(8, 2, 0)
    # test = torch.randn(2, 8, 5, 7)
    # out = atten(test)

    cross = CrossAttention(8, 10, 2, 0)
    test = torch.randn(2, 8, 5, 7)
    label = torch.randn(2, 10)
    out = cross(test, label)

    print('hi')

