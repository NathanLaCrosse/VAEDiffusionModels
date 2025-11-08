import math
from random import random

import numpy as np
import torch
from openpyxl.styles.builtins import output
from sympy import convolution
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import NetworkComponents as nc
import mushroomdata


class UNET(nn.Module):

    def __init__(self, time_embed_dim, label_embed_dim, num_classes, dropout_p=0.0):
        super(UNET, self).__init__()
        #starting with a 8x8x8 latent
        #time_emb is set
        #label_imb is set

        # For creating global time & label vectors
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
            nn.SiLU()
        )
        self.label_mlp = nn.Sequential(
            nn.Embedding(num_classes, label_embed_dim),
            nn.Linear(label_embed_dim, label_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(label_embed_dim*4, label_embed_dim),
            nn.SiLU()
        )

        # Downward pass of the UNet
        self.initial = nn.Conv2d(8, 16, 1) # 8 x 8 x 8 -> 16 x 8 x 8
        self.downres1 = nc.ResidualBlockWithEmbeddings(16, 8, 8, time_embed_dim=time_embed_dim,
                                                   label_embed_dim=label_embed_dim, dropout_p=dropout_p) # 16 x 8 x 8
        self.down1 = nn.Conv2d(16, 32, 3, 2, 1) # 16 x 8 x 8 -> 32 x 4 x 4
        self.downres2 = nc.ResidualBlockWithEmbeddings(32, 16, 4, time_embed_dim=time_embed_dim,
                                                   label_embed_dim=label_embed_dim, dropout_p=dropout_p) # 32 x 4 x 4
        self.down2 = nn.Conv2d(32, 64, 3, 2, 1) # 32 x 4 x 4 -> 64 x 2 x 2
        self.downres3 = nc.ResidualBlockWithEmbeddings(64, 32, 2, time_embed_dim=time_embed_dim,
                                                   label_embed_dim=label_embed_dim, dropout_p=dropout_p) # 64 x 2 x 2

        # Upward pass of the UNet
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, 2) # 64 x 2 x 2 -> 32 x 4 x 4
        # Concatenation happens here -> # 64 x 4 x 4 (in forward method)
        self.reduce_channels1 = nn.Conv2d(64, 32, 1) # 64 x 4 x 4 -> 32 x 4 x 4
        self.upres1 = nc.ResidualBlockWithEmbeddings(32, 16, 4, time_embed_dim=time_embed_dim,
                                                     label_embed_dim=label_embed_dim, dropout_p=dropout_p)

        self.upconv2 = nn.ConvTranspose2d(32, 16, 2, 2) # 32 x 4 x 4 -> 16 x 8 x 8
        # Concatenation happens here -> 32 x 8 x 8
        self.reduce_channels2 = nn.Conv2d(32, 16, 1) # 32 x 8 x 8 -> 16 x 8 x 8
        self.upres2 = nc.ResidualBlockWithEmbeddings(16, 8, 8, time_embed_dim=time_embed_dim,
                                                     label_embed_dim=label_embed_dim, dropout_p=dropout_p)

        self.to_out = nn.Conv2d(16, 8, 1)

    def forward(self, x, time_embed, l):
        global_t = self.time_mlp(time_embed)
        global_l = self.label_mlp(l)

        step1 = self.initial(x) # 8 x 8 x 8 -> 16 x 8 x 8
        step1 = self.downres1(step1, global_t, global_l) # 16 x 8 x 8 -> 16 x 8 x 8

        step2 = self.down1(step1) # 16 x 8 x 8 -> 32 x 4 x 4
        step2 = self.downres2(step2, global_t, global_l) # 32 x 4 x 4 -> 32 x 4 x 4

        step3 = self.down2(step2) # 32 x 4 x 4 -> 64 x 2 x 2
        step3 = self.downres3(step3, global_t, global_l) # 64 x 2 x 2 -> 64 x 2 x 2 (Bottom step)

        up = self.upconv1(step3) # 64 x 2 x 2 -> 32 x 4 x 4
        up = torch.cat([step2, up], dim=1) # 32 x 4 x 4 -> 64 x 4 x 4
        up = self.reduce_channels1(up) # 64 x 4 x 4 -> 32 x 4 x 4
        up = self.upres1(up, global_t, global_l) # 32 x 4 x 4 -> 32 x 4 x 4

        up = self.upconv2(up) # 32 x 4 x 4 -> 16 x 8 x 8
        up = torch.cat([step1, up], dim=1) # 16 x 8 x 8 -> 32 x 8 x 8
        up = self.reduce_channels2(up) # 32 x 8 x 8 -> 16 x 8 x 8
        up = self.upres2(up, global_t, global_l) # 16 x 8 x 8 -> 16 x 8 x 8

        return self.to_out(up)

if __name__ == '__main__':
    test_x = torch.randn(2,8,8,8)
    test_t = torch.randn(2,10)
    test_l = torch.randint(7, (2,))

    unet = UNET(10, 13, 7)

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