import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import mushroomdata
from VAE_128 import VAE
import lpips
import trainunet as u


def train_nn(epochs=15, batch_size=32, lr=0.001, num_periods=5, beta_mult=0.1, percep_mult=1, save_file="mushroom_vae.pt", load_file=None, latent_channels=4, mse_mode=False):
    #Load the picture data
    # dataset = mushroomdata.MushroomData("DataJsons/traindirs.json", mse_mode=mse_mode)
    dataset = mushroomdata.MushroomData("DataJsons/combineddirs.json", True, "MushroomData/")
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}.")

    model = VAE(latent_channels=latent_channels, dropout=0.05).to(device)

    if load_file is not None:
        model.load_state_dict(torch.load(load_file, map_location=device))

    # reconstruction_loss = nn.BCEWithLogitsLoss(reduction="mean")
    reconstruction_loss = nn.MSELoss(reduction="mean")
    # percep_loss = PerceptualLoss()
    percep_loss = lpips.LPIPS(net='alex').to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    beta_multiplier = beta_mult
    period_length = epochs * len(dataloader) / num_periods
    batch_idx = 0

    for epoch in range(epochs):
        p_bar = tqdm(dataloader, desc=f"Epoch [{epoch + 1} / {epochs}]")
        for images in p_bar:
            images = images[0].to(device)

            beta = beta_multiplier * np.sin(np.pi * (batch_idx % period_length) / period_length)

            optimizer.zero_grad()
            outputs, KL_div = model(images)

            # Loss consists of three terms - reconstruction, perceptual, KL-divergence
            r_loss = reconstruction_loss(outputs, images)
            p_loss = percep_loss(outputs, images).mean()
            k_loss = KL_div

            loss = r_loss + p_loss * percep_mult + k_loss * beta
            loss.backward()
            optimizer.step()

            batch_idx += 1
            # Print out unscaled loss values
            p_bar.set_postfix({
                'Recon Loss' : r_loss.item(),
                'Percep Loss' : p_loss.item(),
                'KL Loss' : k_loss.item(),
                'Beta' : beta
            })

        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"PTFiles/inprogress{epoch}{save_file}")
    torch.save(model.state_dict(), f"PTFiles/{save_file}")

    return model

if __name__ == '__main__':
    epochs = 75
    batch_size = 64
    # vae = train_nn(epochs, batch_size, lr=0.001, num_periods=6, beta_mult=0.00001, percep_mult=0.05, save_file="cleaned_vae.pt", load_file=None, latent_channels=8, mse_mode=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(4).to(device)
    vae.load_state_dict(torch.load("PTFiles/vae_128.pt", map_location=device))
    vae = vae.eval()

    u.train_unet(epochs=100, batch_size=64, file_base="unet128.pt", num_time_steps=200, learning_rate=1e-4,
               dropout=0, previous_epochs=0, vae_file=None, latent_width=16, load_file=None, given_vae=vae, num_classes=74)
    # u.train_unet(epochs=150, batch_size=64, file_base="cleaned_unet_steps.pt", num_time_steps=1000, learning_rate=1e-4,
    #            dropout=0, previous_epochs=0, vae_file=None, latent_width=16, load_file=None, given_vae=vae, num_classes=74)


