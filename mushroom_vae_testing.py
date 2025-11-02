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

dat = mushroomdata.MushroomData("DataJsons/testdirs.json")
model = VAE()

model.load_state_dict(torch.load("mushroom_vae.pt"))

with torch.no_grad():
    for im, label in dat:
        pred = model(im.view(1, 3, 64, 64))
        pred = pred[0][0]
        pred = F.sigmoid(pred)

        im = im.permute(1,2,0)
        pred = pred.permute(1,2,0)

        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(im)
        ax[1].imshow(pred)
        plt.show()