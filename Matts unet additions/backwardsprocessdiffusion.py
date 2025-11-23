import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import NetworkComponents as nc
import mushroomdata
from UNetArchitecture import UNET, GeneralizedUNet
from VAE_128 import VAE
import matplotlib.pyplot as plt
from torch_ema import ExponentialMovingAverage
import cv2

import json
import os
import random
from PIL import Image

denoise_steps = 1000
noise_scaling = 1
sample_scaling = 1
time_emb_dim = 128
label_emb_dim = 256
latent_dim = 32
num_classes = 74
latent_channels = 4

cpu = torch.device('cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('DataJsons/idx2class.json', 'r') as file:
    idx2class = json.load(file)

# vae = VAE(8)
# vae.load_state_dict(torch.load("PTFiles/cleaned_vae.pt", map_location=device))

vae = VAE(4).to(device)
vae.load_state_dict(torch.load("PTFiles/vae_128.pt", map_location=device))

# unet = UNET(time_emb_dim, label_emb_dim, num_classes, starting_scale=16)
unet = GeneralizedUNet(time_emb_dim, label_emb_dim, num_classes, 4)
ema = ExponentialMovingAverage(unet.parameters(), decay=0.9999)

# checkpoint = torch.load("PTFiles/cleaned_unet.pt", map_location=device)
checkpoint = torch.load("PTFiles/unet_128_channelsup50percent.pt", map_location=device)
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

time_encodings = nc.positional_encoding(num_time_steps, time_emb_dim).to(device)

stats = torch.load("latent_channel_info.pt")
latent_means = stats['means'].to(device).view(1, latent_channels, latent_dim, latent_dim)
latent_stds = stats['stds'].to(device).view(1, latent_channels, latent_dim, latent_dim)

#Graph components
rows = 3
cols = 3


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
          
def plot_real_mushrooms(labels, mushroom_img_folder = 'CleanedData'):
    real_fig, real_ax = plt.subplots(rows, cols)
    real_fig.suptitle("Real Picture Examples", fontsize=16)

    for i in range(rows):
        for j in range(cols):
            label_idx = labels[i * cols + j].item()
            species_name = idx2class[str(label_idx)]
            species_folder = os.path.join(mushroom_img_folder, species_name)
            image_files = [picture for picture in os.listdir(species_folder) if picture.endswith(".png")]
            random_image = random.choice(image_files)
            img_path = os.path.join(species_folder, random_image)
            img = Image.open(img_path).convert("RGB")

            real_ax[i, j].imshow(img)
            real_ax[i, j].set_title(species_name, fontsize=9)
            real_ax[i, j].axis("off")

    plt.tight_layout()
    plt.show(block=False)

def denoise_step_by_step(latent, unet, alphas, betas, alpha_bars, time_encodings, total_noise_steps, label):
    bs, _, height, width = latent.size()
    pred = latent
    t = total_noise_steps


    with torch.no_grad():
       original_noise  = vae.forward_decode_only(pred)

    def display_images(picture_batch):
        pictures = []
        for i in range(rows):
            row_i_pictures = []
            for j in range(cols):
                picture = picture_batch[i * cols + j].to(cpu)
                picture = (picture + 1) / 2
                picture = picture.permute(1, 2, 0).numpy()
                picture = (picture * 255).astype("uint8")
                picture = cv2.cvtColor(picture, cv2.COLOR_RGB2BGR)

                row_i_pictures.append(picture)
            pictures.append(np.hstack(row_i_pictures))
        return np.vstack(pictures)

    frame = display_images(original_noise)
    plot_real_mushrooms(label)

    title = "Step by step denoising animation"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, frame)
    cv2.waitKey(1)

    with torch.no_grad():
        with ema.average_parameters():
            while t > 0:
                cv2.waitKey(1)
                index = label[0].item()
                species = idx2class[str(index)]
                cv2.setWindowTitle(title, f"Denoising Step {t}: {species}")

                step_vect = time_encodings[t - 1].unsqueeze(0).expand(bs, time_emb_dim)

                noise = unet(pred, step_vect, label) * noise_scaling

                pred = 1 / torch.sqrt(alphas[t - 1]) * (pred - betas[t - 1] / torch.sqrt(1 - alpha_bars[t - 1]) * noise)

                if t > 1:
                    pred = pred + torch.sqrt(betas[t - 1]) * torch.randn_like(pred)

                # decode then plot the decoding step
                if t % 4 == 0:
                    decoded = vae.forward_decode_only(pred)
                    frame = display_images(decoded)
                    cv2.imshow(title, frame)

                #go down one time step
                t  = t-1
    cv2.waitKey(0)
    return pred

def plot_final_result():
    # Actual testing stuff here
    print('Starting...')
    with torch.no_grad():
        while True:

            samp = latent_means + latent_stds * sample_scaling * torch.randn((rows*cols, latent_channels, latent_dim, latent_dim), device=device)
            # samp = torch.randn((rows*cols, 8, 8, 8), device=device)

            labels = torch.randint(0,num_classes,(rows*cols,), device=device)

            denoised = denoise_latent(samp, unet, labels, alphas, betas, alpha_bars, time_encodings, denoise_steps)

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

def plot_denoising_animation():
    with torch.no_grad():
        bs = rows * cols
        samp = latent_means + latent_stds * sample_scaling * torch.randn((rows * cols, latent_channels, latent_dim, latent_dim), device=device)


        #TODO:: Make this a choice of random or select a species
        labels = torch.randint(1, 2, (bs,), device=device)

        #latent, unet, alphas, betas, alpha_bars, time_encodings, total_noise_steps, label
        denoise_step_by_step(samp, unet, alphas, betas, alpha_bars, time_encodings, num_time_steps, labels)


plot_denoising_animation()
# plot_final_result()