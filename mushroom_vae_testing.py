import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import mushroomdata
from VAE_128 import VAE
import matplotlib.pyplot as plt

"""
This file is designed to compare the results of inputted vae models. When training a vae
overnight, we often created a few copies with different hyperparameters, hoping at least
one of them did not experience posterior collapse. We used this file to determine the
best ones.
"""

latent_options = [4]
state_dicts = [
    "PTFiles/vae_128.pt"
]
im_size = 128

models = [
    VAE(latent_channels=latent_options[i]) for i in range(len(latent_options))
]
for i in range(len(latent_options)):
    models[i].load_state_dict(torch.load(state_dicts[i], map_location=torch.device('cpu')))
    models[i] = models[i].eval()

# dat = mushroomdata.MushroomData("DataJsons/testdirs.json", mse_mode=True)
dat = mushroomdata.MushroomData("DataJsons/combineddirs.json", True, "CleanedData/", halve=True)

with torch.no_grad():
    for im, label in dat:
        preds = [None for i in range(len(latent_options))]

        for i in range(len(latent_options)):
            preds[i] = (models[i](im.view(1,3,im_size,im_size))[0][0] + 1) / 2
            preds[i] = preds[i].permute(1, 2, 0)

            # Print latent statistics -> should be about mean 0 std 1
            latent = models[i].forward_encode_only_mean(im.view(1,3,im_size,im_size))
            print(latent.mean().item(), latent.std().item())

        # Results from decoder are between -1 and 1.
        im = (im + 1) / 2
        im = im.permute(1,2,0)

        fig, ax = plt.subplots(1, len(latent_options)+1)

        ax[0].imshow(im)
        ax[0].set_title("Original")

        for i in range(len(latent_options)):
            ax[1+i].imshow(preds[i])
            ax[1+i].set_title(f"Option {1+i}")

        plt.show()