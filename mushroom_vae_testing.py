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

model = VAE(latent_channels=16)

model.load_state_dict(torch.load("beta4.pt"))

mse_mode = False
dat = mushroomdata.MushroomData("DataJsons/testdirs.json", mse_mode=mse_mode)

with torch.no_grad():
    for im, label in dat:
        pred = model(im.view(1, 3, 64, 64))
        pred = pred[0][0]
        if mse_mode:
            pred = (pred + 1) / 2
            im = (im + 1) / 2
        else:
            pred = F.sigmoid(pred)

        im = im.permute(1,2,0)
        pred = pred.permute(1,2,0)

        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(im)
        ax[1].imshow(pred)
        plt.show()