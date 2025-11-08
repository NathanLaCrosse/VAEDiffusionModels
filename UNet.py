import math
from random import random

import numpy as np
import torch
from torch import nn, randn
from torch.utils.data import DataLoader
from tqdm import tqdm

import mushroomdata
from NathanVAE import VAE


class NoiseFinder(nn.Module):

    def __init__(self):
        super(NoiseFinder, self).__init__()

        # Downward pass of the UNet
        ## 8 x 8 x 8
        # self.down1 = ConvolutionBlock() # 16 x 4 x 4
        # self.down2 = ConvolutionBlock() # 32 x 2 x 2
        ## 64 x 1 x 1

        # Upward pass of the UNet
        ## 64 x 1 x 1
        # self.up1 = ConvolutionBlock() # 32 x 2 x 2
        # self.up2 = ConvolutionBlock() # 16 x 4 x 4
        ## 8 x 8 x 8
        # self.upconv1 = nn.ConvTranspose2d(256, 128, 2, 2, 0)
        # self.up1 = ConvolutionBlock([256, 128, 128], 16, 16)
        # self.upconv2 = nn.ConvTranspose2d(128, 64, 2, 2, 0)
        # self.up2 = ConvolutionBlock([128, 64, 64], 32, 32)
        # self.upconv3 = nn.ConvTranspose2d(64, 32, 2, 2, 0)
        # self.up3 = ConvolutionBlock([64, 32, 32], 64, 64)

        # Final 1x1 convolution to map to a single grid
        self.one = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # # Perform downward pass
        # x = self.down1(x)  # 8 x 8 x 8
        # res1 = self.ResNET(x, time, label)
        # x = F.max_pool2d(res1, 2, 2)

        # x = self.down2(x)  # 16 x 4 x 4
        # res2 = self.ResNET(x, time, label)
        # x = F.max_pool2d(res2, 2, 2)

        # x = self.down3(x)  # 32 x 2 x 2
        # res3 = self.ResNET(x, time, label)
        # x = F.max_pool2d(res3, 2, 2)

        # res4 = self.down4(x)  # 64 x 1 x 1

        # Do upward pass, adding in residual skip connections along the way
        # x = self.upconv1(res4)  # 128 x 16 x 16
        # x = torch.cat([res3, x], dim=1)  # 256 x 16 x 16
        # x = self.up1(x)  # 128 x 16 x 16
        #
        # x = self.upconv2(x)  # 64 x 32 x 32
        # x = torch.cat([res2, x], dim=1)  # 128 x 32 x 32
        # x = self.up2(x)  # 64 x 32 x 32
        #
        # x = self.upconv3(x)  # 32 x 64 x 64
        # x = torch.cat([res1, x], dim=1)  # 64 x 64 x 64
        # x = self.up3(x)  # 32 x 64 x 64

        return self.one(x)


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

    alpha_bar = np.prod(alpha_steps)

    unet_model = NoiseFinder()

    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        p_bar = tqdm(dataloader, desc=f"Epoch [{epoch + 1} / {epochs}]")
        for images in p_bar:
            images = images.to(device)
            r_indx = int(random() * num_time_steps)

            #TODO Change to be actual latents
            input_latent = np.array([8,8,8])
            noisy_latents = math.sqrt(alpha_steps[r_indx]) * input_latent + math.sqrt(1 - alpha_bar) *randn()

            optimizer.zero_grad()

            output_latent = unet_model(noisy_latents)

            loss = loss_fn(input_latent, output_latent)
            loss.backward()
            optimizer.step()

        torch.save(unet_model.state_dict(), f"PTFiles/{file_base}")
        if (epoch + 1) % 25 == 0:
            torch.save(unet_model.state_dict(), f"PTFiles/inprogress{epoch}{file_base}")


train_unet()