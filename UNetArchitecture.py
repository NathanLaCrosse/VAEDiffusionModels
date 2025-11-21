import numpy as np
import torch
from sympy import convolution
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import NetworkComponents as nc

class UNET(nn.Module):

    def __init__(self, time_embed_dim, label_embed_dim, num_classes, dropout_p=0.0, starting_scale=8):
        super(UNET, self).__init__()
        #starting with a 8x8x8 latent
        #time_emb is set
        #label_imb is set

        # For creating global time & label vectors
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
            nn.SiLU()
        )
        self.label_embed = nn.Embedding(num_classes, label_embed_dim)
        self.label_mlp = nn.Sequential(
            nn.Linear(label_embed_dim, label_embed_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(label_embed_dim*4, label_embed_dim),
            nn.SiLU()
        )

        # Downward pass of the UNet
        self.initial = nn.Conv2d(8, 48, 1) # 8 x 8 x 8 -> 48 x 8 x 8

        self.pass1 = nc.UNetLayer(48, starting_scale, time_embed_dim, label_embed_dim, dropout_p) # 48 x 8 x 8 retained throughout
        self.down1 = nn.Conv2d(48, 96, 3, 2, 1) # 48 x 8 x 8 -> 96 x 4 x 4

        self.pass2 = nc.UNetLayer(96, starting_scale//2, time_embed_dim, label_embed_dim, dropout_p) # 64 x 4 x 4
        self.down2 = nn.Conv2d(96, 192, 3, 2, 1) # 96 x 4 x 4 -> 192 x 2 x 2

        self.pass3 = nc.UNetLayer(192, starting_scale//4, time_embed_dim, label_embed_dim, dropout_p) # 192 x 2 x 2
        self.up1 = nn.ConvTranspose2d(192, 96, 2, 2) # 192 x 2 x 2 -> 96 x 4 x 4
        # Concatenation here -> 192 x 4 x 4

        self.pass4 = nc.UNetLayer(192, starting_scale//2, time_embed_dim, label_embed_dim, dropout_p) # 192 x 4 x 4
        self.up2 = nn.ConvTranspose2d(192, 48, 2, 2) # 192 x 4 x 4 -> 48 x 8 x 8
        # Concatenation here -> 96 x 8 x 8

        # self.pass5 = nc.NResBlocks(2, 64, 32, 8, time_embed_dim, label_embed_dim, dropout_p)
        self.pass5 = nc.UNetLayer(96, starting_scale, time_embed_dim, label_embed_dim, dropout_p)
        self.to_out = nn.Conv2d(96, 8, 1)



    def forward(self, x, time_embed, l):
        global_t = self.time_mlp(time_embed)
        global_l = self.label_mlp(self.label_embed(l))

        x = self.initial(x)

        res1 = self.pass1(x, global_t, global_l) # 32 x 8 x 8

        res2 = self.down1(res1)
        res2 = self.pass2(res2, global_t, global_l) # 64 x 4 x 4

        res3 = self.down2(res2)
        res3 = self.pass3(res3, global_t, global_l) # 128 x 2 x 2

        up = self.up1(res3) # 64 x 4 x 4
        up = torch.cat([up, res2], dim=1) # 128 x 4 x 4
        up = self.pass4(up, global_t, global_l)

        up = self.up2(up) # 32 x 8 x 8
        up = torch.cat([up, res1], dim=1) # 64 x 8 x 8
        up = self.pass5(up, global_t, global_l)

        return self.to_out(up)


class GeneralizedUNet(nn.Module):

    def __init__(self, time_embed_dim, label_embed_dim, num_classes, down_passes,
                 dropout_p=0.0, starting_scale=32, latent_channels=4):
        super(GeneralizedUNet, self).__init__()

        # For creating global time & label vectors
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
            nn.SiLU()
        )
        self.label_embed = nn.Embedding(num_classes, label_embed_dim)
        self.label_mlp = nn.Sequential(
            nn.Linear(label_embed_dim, label_embed_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(label_embed_dim * 4, label_embed_dim),
            nn.SiLU()
        )

        channel_sizes = [32, 64, 128, 256, 512, 1024, 2048]
        self.initial = nn.Conv2d(latent_channels, channel_sizes[0], 1)
        self.pass1 = nc.UNetLayer(channel_sizes[0], starting_scale, time_embed_dim, label_embed_dim, dropout_p)

        self.down_passes = down_passes
        self.down_steps = nn.ModuleList()

        for i in range(down_passes):
            self.down_steps.append(
                nn.Conv2d(channel_sizes[i], channel_sizes[i+1], 3, 2, 1)
            ) # Half im size, double channels
            self.down_steps.append(
                nc.UNetLayer(channel_sizes[i+1], starting_scale // (2**(i+1)), time_embed_dim, label_embed_dim, dropout_p)
            ) # Retain size, perform residual blocks & attention

        self.up_steps = nn.ModuleList()

        for i in range(down_passes):
            self.up_steps.append(
                nn.ConvTranspose2d(channel_sizes[down_passes-i], channel_sizes[down_passes-i-1], 2, 2)
            ) # Double im size, halve channels
            # Concatenation happens here
            self.up_steps.append(nn.Conv2d(channel_sizes[down_passes-i], channel_sizes[down_passes-i-1], 1)) # Combine skip data
            self.up_steps.append(
                nc.UNetLayer(channel_sizes[down_passes-i-1], starting_scale // (2**(down_passes-i-1)), time_embed_dim, label_embed_dim, dropout_p)
            ) # Retain size, perform residual blocks & attention

        self.to_out = nn.Conv2d(channel_sizes[0], 4, 1)

    def forward(self, x, time_embed, l):
        global_t = self.time_mlp(time_embed)
        global_l = self.label_mlp(self.label_embed(l))

        x = self.initial(x)

        # Collect residuals and do down path
        residuals = [x]
        prev_res = x

        for i in range(self.down_passes):
            res = self.down_steps[i*2](prev_res) # Downsample
            res = self.down_steps[i*2+1](res, global_t, global_l) # Perform residual blocks & attention

            residuals.append(res)
            prev_res = res

        # Perform upward path
        up = prev_res

        for i in range(self.down_passes):
            up = self.up_steps[i*3](up) # Upsampling
            up = torch.cat([up, residuals[-(2+i)]], dim=1) # Concatenation
            up = self.up_steps[i*3+1](up) # Combine skip & down channels
            up = self.up_steps[i*3+2](up, global_t, global_l) # Perform residual blocks

        return self.to_out(up)



if __name__ == "__main__":
    test = torch.randn((3, 4, 32, 32))
    t_vect = torch.randn((3, 128))
    l_vect = torch.randint(0, 74, (3,))

    u = GeneralizedUNet(128, 256, 74, 3)

    out = u(test, t_vect, l_vect)

    print('hi')