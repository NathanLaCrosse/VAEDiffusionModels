import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import mushroomdata
from NathanVAE import VAE
import matplotlib.pyplot as plt
import os

latent_channels = 8
model = VAE(latent_channels=latent_channels)
mse_mode = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

name = "tinypercep1"
model.load_state_dict(torch.load(f"PTfiles/{name}.pt", map_location=device))
model = model.to(device)

train_dat = mushroomdata.MushroomData(json_file="DataJsons/traindirs.json", mse_mode=mse_mode)
batch_size = 128

with torch.no_grad():

    # Calculate (or retrieve) the mean and standard deviation of the whole distribution
    # ---> (over every singe latent vector)
    global_mean = torch.zeros((latent_channels, 8, 8), device=device)
    global_std = torch.zeros((latent_channels, 8, 8), device=device)
    cpu = torch.device('cpu')
    #
    # try:
    #     global_mean = torch.tensor(np.load(f"GlobalMeanStds/g_mean_{name}.npy"))
    #     global_std = torch.tensor(np.load(f"GlobalMeanStds/g_std_{name}.npy"))
    # except OSError:
    #     dat_loader = DataLoader(train_dat, batch_size=batch_size)
    #     progress = tqdm(dat_loader, desc="Computing global mean & std")
    #
    #     for _, batch in enumerate(progress):
    #         ims = batch[0]
    #         ims = ims.to(device)
    #
    #         means, stds = model.forward_to_mean_std(ims)
    #
    #         global_mean += torch.sum(means, dim=0)
    #         global_std += torch.sum(stds, dim=0)
    #
    #     global_mean = global_mean / len(train_dat)
    #     global_std = global_std / len(train_dat)
    #
    #     np.save(f"GlobalMeanStds/g_mean_{name}.npy", global_mean.to(cpu).numpy())
    #     np.save(f"GlobalMeanStds/g_std_{name}.npy", global_std.to(cpu).numpy())
    #
    # global_mean = global_mean.to(cpu)
    # global_std = global_std.to(cpu)
    model = model.to(cpu)

    # Randomly sample from mean and standard deviation
    rows = 5
    cols = 5
    fig, ax = plt.subplots(rows, cols)
    for i in range(rows):
        for j in range(cols):
            # x = global_mean + global_std * torch.randn_like(global_std)
            x = torch.randn_like(global_std)
            x = model.forward_decode_only(x).permute(1, 2, 0)
            if not mse_mode:
                x = F.sigmoid(x)
            else:
                x = (x + 1) / 2
            ax[i, j].imshow(x)

    plt.show()
