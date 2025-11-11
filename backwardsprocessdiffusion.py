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

cpu = torch.device('cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = VAE(8)
unet = UNET(64, 128, 100)

vae.load_state_dict(torch.load("PTFiles/largernorm3.pt", map_location=device))
unet.load_state_dict(torch.load("PTFiles/overnight.pt", map_location=device))
vae = vae.to(device).eval()
unet = unet.to(device).eval()

start_step = 0.0001
end_step = 0.02
num_time_steps = 1000

betas = np.linspace(start_step, end_step, num_time_steps)
alphas = 1 - betas
alpha_bars = np.zeros(num_time_steps)
alpha_bars[0] = alphas[0]
for i in range(1, num_time_steps):
    alpha_bars[i] = alphas[i] * alpha_bars[i-1]
time_encodings = nc.positional_encoding(num_time_steps, 64)

# print(alpha_bars[:5], alpha_bars[-5:])

# Method to decode latent -> formula from class
def denoise_latent(latent, unet, alphas, betas, alpha_bars, time_encodings, total_noise_steps, label):
    bs, _, _, _ = latent.size()
    pred = latent

    t = total_noise_steps
    with torch.no_grad():
        while t > 0:
            step_vect = time_encodings[t-1].unsqueeze(0).expand(bs, 64).to(device)
            noise = unet(pred, step_vect, label)

            pred = 1 / np.sqrt(alphas[t-1]) * (pred - betas[t-1] / np.sqrt(1 - alpha_bars[t-1]) * noise)

            if t > 1:
                pred = pred + np.sqrt(betas[t-1]) * torch.randn_like(pred)

            t -= 1
        
        return pred


stats = torch.load("latent_channel_info.pt")
latent_means = stats['means'].to(device).view(1, -1, 1, 1)
latent_stds = stats['stds'].to(device).view(1, -1, 1, 1)
std_shift = 0.4

# Actual testing stuff here
with torch.no_grad():
    while True:
        rows = 4
        cols = 4

        samp = latent_means + latent_stds * std_shift * torch.randn((rows*cols, 8, 8, 8), device=device)
        labels = torch.randint(0,100,(rows*cols,), device=device)

        denoised = denoise_latent(samp, unet, alphas, betas, alpha_bars, time_encodings, num_time_steps, labels)

        ims = vae.forward_decode_only(denoised)

        # print("Latent stats:", )
        # print("Denoised stats:", denoised[0].mean().item(), denoised.std().item())
        
        # print("Decoded range:", ims[0].min().item(), ims[0].max().item())
        # print("Samples:", ims[0].view(-1)[:10].tolist())

        fig, ax = plt.subplots(rows, cols)

        for i in range(rows):
            for k in range(cols):
                im = ims[i*rows + k].to(cpu)
                im = (im + 1) / 2
                im = im.permute(1,2,0)

                # print("Decoded stats:", im[0].mean().item(), im[0].std().item())

                ax[i, k].imshow(im)
                ax[i, k].axis('off')
        
        plt.tight_layout()
        plt.show()