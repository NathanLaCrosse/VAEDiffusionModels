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

denoise_steps = 1000
noise_scaling = 1
sample_scaling = 1

cpu = torch.device('cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = VAE(8)
unet = UNET(128, 256, 100)
ema = ExponentialMovingAverage(unet.parameters(), decay=0.9999)

vae.load_state_dict(torch.load("PTFiles/largernorm3.pt", map_location=device))

checkpoint = torch.load("PTFiles/more_channels.pt", map_location=device)
unet.load_state_dict(checkpoint['model'])
ema.load_state_dict(checkpoint['ema'])

vae = vae.to(device).eval()
unet = unet.to(device).eval()

start_step = 0.0001
end_step = 0.02
num_time_steps = 1000

betas = torch.linspace(start_step, end_step, num_time_steps, device=device)
alphas = 1 - betas
alpha_bars = torch.zeros(num_time_steps, device=device)
alpha_bars[0] = alphas[0]
for i in range(1, num_time_steps):
    alpha_bars[i] = alphas[i] * alpha_bars[i-1]

time_encodings = nc.positional_encoding(num_time_steps, 128).to(device)

stats = torch.load("latent_channel_info.pt")
latent_means = stats['means'].to(device).view(1, 8, 8, 8)
latent_stds = stats['stds'].to(device).view(1, 8, 8, 8)

# Method to decode latent -> formula from class
def denoise_latent(latent, unet, labels, alphas, betas, alpha_bars, time_encodings, total_noise_steps):
    bs, _, _, _ = latent.size()
    pred = latent

    t = total_noise_steps
    with torch.no_grad():
        with ema.average_parameters():
            while t > 0:
                step_vect = time_encodings[t-1].unsqueeze(0).expand(bs, 128)
                
                noise = unet(pred, step_vect, labels) * noise_scaling

                pred = 1 / torch.sqrt(alphas[t-1]) * (pred - betas[t-1] / torch.sqrt(1 - alpha_bars[t-1]) * noise)

                if t > 1:
                    pred = pred + torch.sqrt(betas[t-1]) * torch.randn_like(pred)

                t -= 1
            
            return pred


# Actual testing stuff here
print('Starting...')
with torch.no_grad():
    while True:
        rows = 7
        cols = 7

        samp = latent_means + latent_stds * sample_scaling * torch.randn((rows*cols, 8, 8, 8), device=device)
        # samp = torch.randn((rows*cols, 8, 8, 8), device=device)
        #
        labels = torch.randint(0,1,(rows*cols,), device=device)

        denoised = denoise_latent(samp, unet, labels, alphas, betas, alpha_bars, time_encodings, denoise_steps)
        # denoised = denoise_latent_ddim(samp, unet, alpha_bars, time_encodings, 1000, 250, 0.2, ema, device)

        # denoised = denoised * latent_stds + latent_means

        ims = vae.forward_decode_only(denoised)

        fig, ax = plt.subplots(rows, cols)

        for i in range(rows):
            for k in range(cols):
                im = ims[i*cols + k].to(cpu)
                im = (im + 1) / 2
                im = im.permute(1,2,0)

                ax[i, k].imshow(im)
                ax[i, k].axis('off')
        
        plt.tight_layout()
        plt.show()