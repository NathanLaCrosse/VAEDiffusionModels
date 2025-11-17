import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import NetworkComponents as nc
import mushroomdata
from unet import UNET
from NathanVAE import VAE
import matplotlib.pyplot as plt
from torch_ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
import math

# def cosine_beta_schedule(timesteps, s=0.008):
#     steps = timesteps + 1
#     x = torch.linspace(0, timesteps, steps)
#     alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     return torch.clip(betas, 0.0001, 0.9999)

def cosine_beta_schedule(timesteps, dummy=0.008, device=torch.device('cpu')):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, device=device)
    alphas_cumprod = torch.cos(((x / timesteps) + dummy) / (1 + dummy) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # normalize to start at 1

    # Compute betas from consecutive alpha_bar ratios
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 1e-8, 0.999)  # numerical stability

    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    return betas, alphas, alpha_bars

# x = torch.linspace(0, 1000, 1000, device=device)
x = torch.arange(1, 1001)
betas, alphas, alpha_bars = cosine_beta_schedule(1000)

fig, ax = plt.subplots(3)
ax[0].plot(x, betas)
ax[1].plot(x, alphas)
ax[2].plot(x, alpha_bars)

plt.show()