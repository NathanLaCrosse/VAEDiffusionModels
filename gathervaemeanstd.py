import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import NetworkComponents as nc
import mushroomdata
from VAE_128 import VAE
import matplotlib.pyplot as plt

"""
This file is designed to compute average mean and standard deviations of every single
latent "distribution" in a vae's latent space. This data is used to determine the best
distribution to sample from when generating images.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(4)
vae.load_state_dict(torch.load("PTFiles/vae_128.pt", map_location=device))
vae = vae.to(device).eval()
dat = mushroomdata.MushroomData("DataJsons/combineddirs.json", True, "CleanedData/", halve=True)

batch_size = 128
dat_loader = DataLoader(dat, batch_size=batch_size)
dim = 32

means = torch.zeros((4,dim,dim), device=device) # means over channels
stds = torch.zeros((4,dim,dim), device=device) # means over stds

count = 0
with torch.no_grad():
    p_bar = tqdm(dat_loader, desc="Calculating mean & std...")
    for _, batch in enumerate(p_bar):
        ims, _ = batch
        ims = ims.to(device)

        latents = vae.forward_encode_only_mean(ims)

        if len(ims) > 1:
            means += latents.mean(dim=0)
            stds += latents.std(dim=0)

        count += 1

# Mean the means & stds
means = (means / count).cpu()
stds = (stds / count).cpu()

print("Means:", means)
print("Stds:", stds)

# Load into a special file
torch.save({'means':means, 'stds':stds}, "latent_channel_info.pt")

