import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import mushroomdata
from Matt_VAE import VAE
import matplotlib.pyplot as plt
import os

latent_channels = 8
model = VAE(latent_channels=latent_channels)
mse_mode = True


name = "attn_vae_64x64"
model.load_state_dict(torch.load(f"PTfiles/{name}.pt", map_location=torch.device('cpu')))

train_dat = mushroomdata.MushroomData(json_file="DataJsons/traindirs.json", mse_mode=mse_mode)
batch_size = 128

with torch.no_grad():

    # Calculate (or retrieve) the mean and standard deviation of the whole distribution
    # ---> (over every singe latent vector)
    global_mean = torch.zeros((latent_channels, 16, 16))
    global_std = torch.zeros((latent_channels, 16, 16))
    cpu = torch.device('cpu')

    model = model.eval()

    # Randomly sample from mean and standard deviation
    rows = 5
    cols = 5
    fig, ax = plt.subplots(rows, cols)
    for i in range(rows):
        for j in range(cols):
            x = torch.randn_like(global_std.view(1,latent_channels,16,16))
            x = model.forward_decode_only(x)[0, :, :, :].permute(1, 2, 0)
            if not mse_mode:
                x = F.sigmoid(x)
            else:
                x = (x + 1) / 2
            ax[i, j].imshow(x)

    plt.show()
