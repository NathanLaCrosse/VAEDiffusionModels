import math
import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Implements a self-attention mechanism for use in a convolutional neural network.
    Converts an image of shape (B, C, H, W) into the shape (B, H*W, C) over which
    a self-attention mechanism is applied. Essentially, this block allows all pixels
    to talk with each other. Image is then reshaped back.
    """
    def __init__(self, input_channels, heads=4, dropout=0.0):
        super(MultiHeadSelfAttention, self).__init__()

        self.pos_cache = None
        self.atten = MultiHeadedAttention(input_channels, heads, dropout=dropout)

    def forward(self, x):
        # X Shape: (B, C, H, W)
        batches, channels, rows, cols = x.size()

        # Reshape X to be of size (B, H*W, C)
        x = x.view(batches, channels, rows * cols) # -> (B, C, H*W)
        x = x.permute(0, 2, 1).contiguous() # -> (B, H*W, C)

        # Get positional encoding
        if self.pos_cache is None or self.pos_cache.shape[0] != rows*cols:
            self.pos_cache = two_dimensional_positional_encoding(rows, cols, channels).unsqueeze(0).to(x.device)

        # Apply the attention mechanism
        x = self.atten(x, self.pos_cache)

        # Reshape X back to size (B, C, H, W)
        x = x.permute(0, 2, 1) # -> (B, C, H*W)
        x = x.view(batches, channels, rows, cols).contiguous() # -> (B, C, H, W)

        return x

class CrossAttention(nn.Module):
    """
    Implements a cross-attention mechanism to be applied in a convolutional neural network.
    An image is converted from the shape (B, C, H, W) to (B, H*W, C), which are projected
    into queries for the attention mechanism. A label vector of size (B, L) is linearly
    projected into the key and value vectors. Image is then reshaped back.
    """
    def __init__(self, channel_count, label_embed_size, num_heads, dropout):
        super(CrossAttention, self).__init__()
        self.pos_cache = None

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
        batches, channels, rows, cols = x.size()

        # Reshape X: (B, C, H, W) -> (B, H*W, C)
        x = x.view(batches, channels, rows*cols).permute(0, 2, 1).contiguous()

        # Pre-norm for stabilization
        normed = self.layer_norm1(x)

        # Add positional encoding
        if self.pos_cache is None or self.pos_cache.shape[0] != rows * cols:
            self.pos_cache = two_dimensional_positional_encoding(rows, cols, channels).unsqueeze(0).to(x.device)
        normed = normed + self.pos_cache

        # Use label to generate keys and values
        label_emb = label_emb.unsqueeze(1) # (B, 1, label_embed_size)
        embedded = self.label_projection(label_emb) # (B, 1, channel_count)
        K = self.key_projection(embedded).view(batches, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_projection(embedded).view(batches, 1, self.num_heads, self.head_dim).transpose(1, 2)

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

    def forward(self, input_embeddings, positional_encoding):
        """
        Args:
            input_embeddings (Tensor): shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, sequence_length, embedding_dim = input_embeddings.shape
        num_heads = self.num_heads
        head_dim = self.head_dim

        h = self.layer_norm1(input_embeddings) # Pre-computation norm -> more stable gradients
        h = h + positional_encoding

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

def two_dimensional_positional_encoding(im_rows, im_cols, encoding_dim):
    seq_len = im_rows * im_cols
    rows = torch.arange(seq_len) # Shape (seq_len) -> [0, 1, ..., seq_len-1]
    rows = (rows / im_cols).floor().unsqueeze(1) # (seq_len, 1)

    cols = torch.tile(torch.arange(im_cols), (im_rows,)).unsqueeze(1) # Shape (seq_len, 1)

    i = torch.arange(encoding_dim).unsqueeze(0) # Shape: (1, encoding_dim)
    omega = 1 / torch.pow(10000, (2 * (i // 2)) / encoding_dim) # Shape: (1, encoding_dim)

    row_angles = rows * omega
    col_angles = cols * omega

    pos_enc = torch.zeros(seq_len, encoding_dim)
    midpoint = encoding_dim // 2

    pos_enc[:, 0:midpoint:2] = torch.sin(row_angles[:, 0:midpoint:2])
    pos_enc[:, 1:midpoint:2] = torch.cos(row_angles[:, 1:midpoint:2])
    pos_enc[:, midpoint::2] = torch.sin(col_angles[:, 0:midpoint:2])
    pos_enc[:, midpoint+1::2] = torch.cos(col_angles[:, 1:midpoint:2])

    return pos_enc