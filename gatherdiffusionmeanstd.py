import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import NetworkComponents as nc
import mushroomdata
from trainunet import UNET
from Matt_VAE import VAE
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(8)
vae.load_state_dict(torch.load("PTFiles/cleaned_vae.pt", map_location=device))
vae = vae.to(device).eval()
# dat = mushroomdata.MushroomData("DataJsons/testdirs.json", mse_mode=True)
dat = mushroomdata.MushroomData("DataJsons/combineddirs.json", True, "MushroomData/")

batch_size = 32
dat_loader = DataLoader(dat, batch_size=batch_size)
dim = 16

means = torch.zeros((8,dim,dim), device=device) # means over channels
stds = torch.zeros((8,dim,dim), device=device) # means over stds

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

