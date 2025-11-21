import math
from random import random

import numpy as np
import torch
from sympy import convolution
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import NetworkComponents as nc
import mushroomdata
from Matt_VAE import VAE
from torch.utils.data import WeightedRandomSampler

from torch_ema import ExponentialMovingAverage
from UNetArchitecture import GeneralizedUNet


def train_unet(epochs=15, batch_size = 32, learning_rate = 0.001, num_time_steps = 1000, file_base = "unet.pt",
               vae_file = "PTFiles/largernorm3.pt", vae_latent_channels=8, dropout=0.0, load_file=None, previous_epochs=0,
               warmup_steps=2500, latent_width = 8, given_vae=None, num_classes=100, down_passes=4):
    dataset = mushroomdata.MushroomData("DataJsons/combineddirs.json", True, "MushroomData/")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}.")
    
    if given_vae is None:
        vae_model = VAE(latent_channels=vae_latent_channels)
        vae_model.load_state_dict(torch.load(vae_file, map_location=device))
        vae_model = vae_model.to(device=device)
        vae_model.eval()
    else:
        vae_model = given_vae.eval()

    # Parameter freeze - incredibly important!!!
    for p in vae_model.parameters():
        p.requires_grad = False

    start_step = 0.0001
    end_step = 0.02
    betas = torch.linspace(start_step, end_step, num_time_steps, device=device)
    alphas = 1 - betas

    alpha_bars = torch.zeros(num_time_steps, device=device)
    alpha_bars[0] = alphas[0]
    for i in range(1, num_time_steps):
        alpha_bars[i] = alphas[i] * alpha_bars[i-1]

    # unet_model = UNET(128, 256, num_classes, dropout_p=dropout, starting_scale=16).to(device)
    unet_model = GeneralizedUNet(128, 256, num_classes, down_passes)
    ema = ExponentialMovingAverage(unet_model.parameters(), decay=0.9999)
    ema.to(device)

    if load_file is not None:
        checkpoint = torch.load(load_file, map_location=device)
        unet_model.load_state_dict(checkpoint['model'])
        ema.load_state_dict(checkpoint['ema'])

    time_encodings = nc.positional_encoding(num_time_steps, 128).to(device=device) # (num_time_steps, 64) array of time encodings

    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=learning_rate)

    total_steps = epochs * len(dataset) / batch_size
    min_lr_ratio = 0.02
    already_done_steps = previous_epochs * len(dataset) / batch_size

    def lr_lambda(step):
        step = step + already_done_steps
        # Warmup -> ramp up to base learning rate
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        # Slow decay following a cosine term
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))

        # Make sure we don't go lower than min_lr_ratio
        return max(cosine * (1 - min_lr_ratio) + min_lr_ratio, min_lr_ratio)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    class_weights = class_weights ** 0.5
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    for epoch in range(previous_epochs, epochs):
        dataloader = DataLoader(dataset, batch_size, sampler=sampler)
        p_bar = tqdm(dataloader, desc=f"Epoch [{epoch + 1} / {epochs}]")

        for _, batch in enumerate(p_bar):
            local_bs = len(batch[0])
            ims, labels = batch
            ims = ims.to(device)
            labels = labels.to(device)

            # time_step = np.random.randint(num_time_steps) + 1
            time_steps = torch.randint(0,num_time_steps,(local_bs,),device=device)

            # Generate latents
            latents = vae_model.forward_encode_only_mean(ims).detach()

            # Generate noise and create noisy latents
            noise = torch.randn((local_bs, vae_latent_channels, latent_width, latent_width), device=device)
            used_alpha_bars = alpha_bars[time_steps].view(-1, 1, 1, 1)

            noisy_latents = torch.sqrt(used_alpha_bars) * latents + torch.sqrt(1 - used_alpha_bars) * noise

            optimizer.zero_grad()

            output_latent = unet_model(noisy_latents, time_encodings[time_steps], labels)

            loss = loss_fn(noise, output_latent)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet_model.parameters(), 5.0)
            optimizer.step()
            ema.update()
            scheduler.step()

            p_bar.set_postfix({
                'Loss' : loss.item(),
                'LR' : scheduler.get_last_lr()
            })

        # scheduler.step()
        torch.save({'model' : unet_model.state_dict(), 'ema' : ema.state_dict()}, f"PTFiles/{file_base}")
        if (epoch + 1) % 30 == 0:
            torch.save({'model' : unet_model.state_dict(), 'ema' : ema.state_dict()}, f"PTFiles/inprogress{epoch}{file_base}")

if __name__ == '__main__':
    # train_unet(epochs=50, batch_size=64, file_base="attention.pt", num_time_steps=1000, learning_rate=1e-4, dropout=0.1)
    # train_unet(epochs=50, batch_size=64, file_base="attention1.pt", num_time_steps=1000, learning_rate=1e-4, dropout=0.1, load_file="PTFiles/attention.pt")
    # train_unet(epochs=50, batch_size=64, file_base="attention2.pt", num_time_steps=1000, learning_rate=5e-5, dropout=0.1, load_file="PTFiles/attention1.pt")

    # train_unet(epochs=150, batch_size=64, file_base="attention3.pt", num_time_steps=1000, learning_rate=5e-5, dropout=0.1, load_file="PTFiles/attention2.pt")
    # train_unet(epochs=250, batch_size=64, file_base="deeper_atten2.pt", num_time_steps=1000, learning_rate=1e-4, dropout=0.1, load_file="PTFiles/deeper_atten2.pt", previous_epochs=190)
    # train_unet(epochs=150, batch_size=64, file_base="deeper_atten3.pt", num_time_steps=1000, learning_rate=3e-5, dropout=0.1, load_file="PTFiles/deeper_atten2.pt", previous_epochs=0, warmup_steps=0)


    # train_unet(epochs=200, batch_size=64, file_base="more_channels.pt", num_time_steps=1000, learning_rate=1e-4, dropout=0, previous_epochs=110, load_file="PTFiles/more_channels.pt")

    train_unet(epochs=200, batch_size=64, file_base="new_decoder_unet2.pt", num_time_steps=1000, learning_rate=1e-4,
               dropout=0, previous_epochs=160, vae_file="PTFiles/attn_vae_64x64.pt", latent_width=16, load_file="PTFiles/new_decoder_unet.pt")
    train_unet(epochs=200, batch_size=64, file_base="new_decoder_unetref.pt", num_time_steps=1000, learning_rate=3e-5,
               dropout=0, previous_epochs=52, vae_file="PTFiles/attn_vae_64x64.pt", latent_width=16, load_file="PTFiles/new_decoder_unet.pt", warmup_steps=0)
