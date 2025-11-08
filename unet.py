import math
from random import random

import numpy as np
import torch
from openpyxl.styles.builtins import output
from sympy import convolution
from torch import nn, randn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import NetworkComponents as nc
import mushroomdata


class UNET(nn.Module):

    def __init__(self, t, l):
        super(UNET, self).__init__()
        #starting with a 8x8x8 latent
        #time_emb is set
        #label_imb is set


        # Downward pass of the UNet
        self.res1 = ResidualBlockWithEmbeddings(8, 4, 8, time_embed_dim=t, label_embed_dim=l, dropout_p=0)
        self.down1 = ConvolutionBlock([8,32,32], 4,4) #8x8x8 -> 32x4x4
        self.res2 = ResidualBlockWithEmbeddings(32, 8, 4, time_embed_dim=t, label_embed_dim=l, dropout_p=0)
        self.down2 = ConvolutionBlock([32, 64, 64], 2, 2) # 32x4x4 -> 64x2x2

        #Middle Section (convolution on itself for "global attention"
        self.middle = nn.Conv2d(64, 64, kernel_size=1) #64x2x2

        # Upward pass of the UNet
        self.upconv1 = nn.ConvTranspose2d(64, 32, 3, 2, 1) #64x2x2 -> 32x4x4
        self.up_map1 = nn.Conv2d(64, 32, kernel_size=1, stride=1) #Maps concatinated channels to original size

        ## Not sure if this needs to be here, I don't know really what this adds
        self.covblock_up1 = ConvolutionBlock([32, 16, 32], 4, 4)

        self.upconv2 = nn.ConvTranspose2d(32, 8, 2, 2, 0) #32x4x4 -> 8x8x8
        self.up_map2 = nn.Conv2d(16, 8, kernel_size=1, stride=1) # 16x8x8 -> 8x8x8

    def forward(self, x):
        # Perform downward pass
        res1 = self.down1(x) # 8x8x8 -> 32x4x4
        x = F.max_pool2d(res1, 2, 2)
        res2 = self.down2(x) # 32x4x4 -> 64x2x2

        #Perform Middle convolution
        res2 = self.middle(res2)

        # Do upward pass, adding in residual skip connections along the way
        x = self.upconv1(res2) # 64x2x2 -> 32x4x4
        x = torch.cat([res2, x], dim=1) # 64x4x4
        x = self.up_map1(x) # 64x4x4 -> 32x4x4

        x = self.convblock_up1(x) #32x4x4 -> 32x4x4

        x = self.upconv2(x) #32x4x4 -> 8x8x8
        x = torch.cat([res1, x], dim=1) # 16x8x8
        x = self.up_map2(x) # 16x8x8 -> 8x8x8

        return x


class ConvolutionBlock(nn.Module):
    """
    Does not change size of image, only changes channel counts
    """

    def __init__(self, channel_sequence, rows, cols):
        super(ConvolutionBlock, self).__init__()

        self.conv1 = nn.Conv2d(channel_sequence[0], channel_sequence[1], 3, 1, 1)
        self.conv2 = nn.Conv2d(channel_sequence[1], channel_sequence[2], 3, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


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
            nn.Linear(time_embed_dim*4, bottleneck_channels),
            nn.SiLU()
        )

        # Similar to the time mlp, this draws out locally important information of out the
        # global label information
        self.label_mlp = nn.Sequential(
            nn.Linear(label_embed_dim, label_embed_dim),
            nn.SiLU(),
            nn.Linear(label_embed_dim, im_dim**2),
            nn.SiLU()
        )

        self.conv1 = nn.Conv2d(initial_channels, bottleneck_channels, 1)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(bottleneck_channels, initial_channels, 1)

        # Extra normalization stuff
        self.norm1 = nn.GroupNorm(bottleneck_channels // 4, bottleneck_channels)
        self.norm2 = nn.GroupNorm(initial_channels // 4, initial_channels)

    def forward(self, x, t_vect, l_vect):
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
        local_l = self.label_mlp(l_vect)

        # First convolution
        res = F.silu(self.norm1(self.conv1(x)))

        # local_t has length bottleneck channels -> convert into a view so that
        # it can be added to res
        res = res + local_t[:, :, None, None]

        # local_l has length dim**2 -> convert into a different view added to resp
        local_l = local_l.view(batch_size, self.im_dim, self.im_dim)
        res = res + local_l[:, None, :, :]

        # Perform the rest of the convolutions
        res = F.silu(self.conv2(res))
        res = self.norm2(F.silu(self.conv3(res)))

        if self.dropout_p > 0:
            res = F.dropout(res, p=self.dropout_p)

        # Residual connection
        return F.silu(x + res)

def train_unet(epochs=15, batch_size = 32, learning_rate = 0.1, num_time_steps = 1000, file_base = "unet.pt", vae_file = ""):
    dataset = mushroomdata.MushroomData("DataJsons/traindirs.json")
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}.")

    vae_model = VAE()
    if not vae_file == "":
        vae_model = vae_model.load_state_dict(torch.load(vae_file))

    start_step = 0.001
    end_step = 0.02
    beta_steps = np.array([start_step + (end_step - start_step)*i/(num_time_steps-1) for i in range(num_time_steps)])
    alpha_steps = 1 - beta_steps

    unet_model = UNET()

    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        p_bar = tqdm(dataloader, desc=f"Epoch [{epoch + 1} / {epochs}]")
        for images in p_bar:
            images = images.to(device)
            r_t_indx = int(random() * num_time_steps)

            #TODO Change to be actual latents
            input_latent = np.array([8,8,8])
            rand_epsilon = randn()
            alpha_bar = np.prod(alpha_steps[:r_t_indx])

            noisy_latents = math.sqrt(alpha_bar) * input_latent + math.sqrt(1 - alpha_bar) * rand_epsilon

            optimizer.zero_grad()

            output_latent = unet_model(noisy_latents)

            loss = loss_fn(input_latent, output_latent)
            loss.backward()
            optimizer.step()

        torch.save(unet_model.state_dict(), f"PTFiles/{file_base}")
        if (epoch + 1) % 25 == 0:
            torch.save(unet_model.state_dict(), f"PTFiles/inprogress{epoch}{file_base}")


train_unet()