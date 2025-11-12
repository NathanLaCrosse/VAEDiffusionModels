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
from NathanVAE import VAE
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch_ema import ExponentialMovingAverage

class UNET(nn.Module):

    def __init__(self, time_embed_dim, label_embed_dim, num_classes, dropout_p=0.0):
        super(UNET, self).__init__()
        #starting with a 8x8x8 latent
        #time_emb is set
        #label_imb is set

        # For creating global time & label vectors
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
            nn.SiLU()
        )
        # self.label_mlp = nn.Sequential(
        #     nn.Embedding(num_classes, label_embed_dim),
        #     nn.Linear(label_embed_dim, label_embed_dim * 4),
        #     nn.SiLU(),
        #     nn.Dropout(dropout_p),
        #     nn.Linear(label_embed_dim*4, label_embed_dim),
        #     nn.SiLU()
        # )

        # Downward pass of the UNet
        self.initial = nn.Conv2d(8, 32, 1) # 8 x 8 x 8 -> 16 x 8 x 8
        self.downres1 = nc.NResBlocks(2, 32, 16, 8, time_embed_dim=time_embed_dim, label_embed_dim=label_embed_dim, dropout_p=dropout_p) # 16 x 8 x 8
        
        self.down1 = nn.Conv2d(32, 64, 3, 2, 1) # 16 x 8 x 8 -> 32 x 4 x 4
        self.downres2 = nc.NResBlocks(2, 64, 32, 4, time_embed_dim=time_embed_dim, label_embed_dim=label_embed_dim, dropout_p=dropout_p) # 32 x 4 x 4

        self.down2 = nn.Conv2d(64, 128, 3, 2, 1) # 32 x 4 x 4 -> 64 x 2 x 2
        self.downres3 = nc.NResBlocks(2, 128, 64, 2, time_embed_dim=time_embed_dim, label_embed_dim=label_embed_dim, dropout_p=dropout_p) # 64 x 2 x 2

        # Upward pass of the UNet
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, 2) # 64 x 2 x 2 -> 32 x 4 x 4
        self.smoothing1 = nc.ResidualBlockWithEmbeddings(64, 32, 4, time_embed_dim=time_embed_dim,
                                                         label_embed_dim=label_embed_dim, dropout_p=dropout_p)
        # Concatenation happens here -> # 64 x 4 x 4 (in forward method)
        self.reduce_channels1 = nn.Conv2d(128, 64, 1) # 64 x 4 x 4 -> 32 x 4 x 4
        self.upres1 = nc.NResBlocks(2, 64, 32, 4, time_embed_dim=time_embed_dim, label_embed_dim=label_embed_dim, dropout_p=dropout_p)

        self.upconv2 = nn.ConvTranspose2d(64, 32, 2, 2) # 32 x 4 x 4 -> 16 x 8 x 8
        # Concatenation happens here -> 32 x 8 x 8
        self.smoothing2 = nc.ResidualBlockWithEmbeddings(32, 16, 8, time_embed_dim=time_embed_dim,
                                                         label_embed_dim=label_embed_dim, dropout_p=dropout_p)
        self.reduce_channels2 = nn.Conv2d(64, 32, 1) # 32 x 8 x 8 -> 16 x 8 x 8
        self.upres2 = nc.NResBlocks(2, 32, 16, 8, time_embed_dim=time_embed_dim, label_embed_dim=label_embed_dim, dropout_p=dropout_p)

        self.to_out = nn.Conv2d(32, 8, 1)

    def forward(self, x, time_embed):
        global_t = self.time_mlp(time_embed * 10)
        # global_l = self.label_mlp(l)

        step1 = self.initial(x) # 8 x 8 x 8 -> 16 x 8 x 8
        step1 = self.downres1(step1, global_t) # 16 x 8 x 8 -> 16 x 8 x 8

        step2 = self.down1(step1) # 16 x 8 x 8 -> 32 x 4 x 4
        step2 = self.downres2(step2, global_t) # 32 x 4 x 4 -> 32 x 4 x 4

        step3 = self.down2(step2) # 32 x 4 x 4 -> 64 x 2 x 2
        step3 = self.downres3(step3, global_t) # 64 x 2 x 2 -> 64 x 2 x 2 (Bottom step)

        up = self.smoothing1(self.upconv1(step3), global_t) # 64 x 2 x 2 -> 32 x 4 x 4
        up = torch.cat([step2, up], dim=1) # 32 x 4 x 4 -> 64 x 4 x 4
        up = self.reduce_channels1(up) # 64 x 4 x 4 -> 32 x 4 x 4
        up = self.upres1(up, global_t) # 32 x 4 x 4 -> 32 x 4 x 4

        up = self.smoothing2(self.upconv2(up), global_t) # 32 x 4 x 4 -> 16 x 8 x 8
        up = torch.cat([step1, up], dim=1) # 16 x 8 x 8 -> 32 x 8 x 8
        up = self.reduce_channels2(up) # 32 x 8 x 8 -> 16 x 8 x 8
        up = self.upres2(up, global_t) # 16 x 8 x 8 -> 16 x 8 x 8

        return self.to_out(up)

# def cosine_beta_schedule(timesteps, dummy=0.008, device=torch.device('cpu')):
#     steps = timesteps + 1
#     x = torch.linspace(0, timesteps, steps, device=device)
#     alphas_cumprod = torch.cos(((x / timesteps) + dummy) / (1 + dummy) * math.pi / 2) ** 2
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # normalize to start at 1

#     # Compute betas from consecutive alpha_bar ratios
#     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     betas = torch.clip(betas, 1e-8, 0.999)  # numerical stability

#     alphas = 1.0 - betas
#     alpha_bars = torch.cumprod(alphas, dim=0)

#     return betas, alphas, alpha_bars

def train_unet(epochs=15, batch_size = 32, learning_rate = 0.001, num_time_steps = 1000, file_base = "unet.pt",
               vae_file = "PTFiles/largernorm3.pt", vae_latent_channels=8, dropout=0.0, load_file=None):
    dataset = mushroomdata.MushroomData("DataJsons/traindirs.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}.")

    vae_model = VAE(latent_channels=vae_latent_channels)
    vae_model.load_state_dict(torch.load(vae_file, map_location=device))
    vae_model = vae_model.to(device=device)
    vae_model.eval()

    # Parameter freeze - incredibly important!!!
    for p in vae_model.parameters():
        p.requires_grad = False

    start_step = 0.0001
    end_step = 0.02
    # beta_steps = np.array([start_step + (end_step - start_step)*i/(num_time_steps-1) for i in range(num_time_steps)])

    betas = torch.linspace(start_step, end_step, num_time_steps, device=device)
    alphas = 1 - betas

    alpha_bars = torch.zeros(num_time_steps, device=device)
    alpha_bars[0] = alphas[0]
    for i in range(1, num_time_steps):
        alpha_bars[i] = alphas[i] * alpha_bars[i-1]
    
    # betas, alphas, alpha_bars = cosine_beta_schedule(num_time_steps)
    # alpha_bars = alpha_bars.to(device)

    unet_model = UNET(64, 128, 100, dropout_p=dropout).to(device)
    ema = ExponentialMovingAverage(unet_model.parameters(), decay=0.999)
    ema.to(device)

    if load_file is not None:
        checkpoint = torch.load("PTFiles/ema_deeper.pt", map_location=device)
        unet_model.load_state_dict(checkpoint['model'])
        ema.load_state_dict(checkpoint['ema'])
        # unet_model.load_state_dict(torch.load(load_file, map_location=device))
    # unet_model = unet_model.to(device=device)

    time_encodings = nc.positional_encoding(num_time_steps, 64).to(device=device) # (num_time_steps, 64) array of time encodings

    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=learning_rate)

    # scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    stats = torch.load("latent_channel_info.pt")
    latent_means = stats['means'].to(device).view(1, -1, 1, 1)
    latent_stds = stats['stds'].to(device).view(1, -1, 1, 1)

    for epoch in range(epochs):
        dataloader = DataLoader(dataset, batch_size, shuffle=True)
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
            noise = torch.randn((local_bs, 8, 8, 8), device=device)
            # used_alpha_bars = torch.tensor(alpha_bars, dtype=torch.float32, device=device)[time_steps].view(-1, 1, 1, 1)
            used_alpha_bars = alpha_bars[time_steps].view(-1, 1, 1, 1)

            noisy_latents = torch.sqrt(used_alpha_bars) * latents + torch.sqrt(1 - used_alpha_bars) * noise

            optimizer.zero_grad()

            output_latent = unet_model(noisy_latents, time_encodings[time_steps])

            loss = loss_fn(noise, output_latent)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet_model.parameters(), 5.0)
            optimizer.step()
            ema.update()
            # scheduler.step()

            p_bar.set_postfix({
                'Loss' : loss.item()
            })

        # scheduler.step()
        torch.save({'model' : unet_model.state_dict(), 'ema' : ema.state_dict()}, f"PTFiles/{file_base}")
        if (epoch + 1) % 35 == 0:
            torch.save({'model' : unet_model.state_dict(), 'ema' : ema.state_dict()}, f"PTFiles/inprogress{epoch}{file_base}")

if __name__ == '__main__':
    # batch size could be too big (256 -> 64 -> 32?)
    # train on just the mu, not mu + std (Done!)
    # In diffusion process - sample from distribution generated by means (latent vectors)
    # Can select another scaling factor ~0.4 multiplied by the latent standard deviation
    # Gets closer convergence to the mean
    # train_unet(epochs=200, batch_size=64, file_base="ema_deeper.pt", num_time_steps=1000, learning_rate=1e-4, dropout=0.0)
    # train_unet(epochs=100, batch_size=64, file_base="ema_deeperef.pt", num_time_steps=1000, learning_rate=5e-5, dropout=0.0, load_file="PTFiles/ema_deeper.pt")
    # train_unet(epochs=60, batch_size=32, file_base="ema_deeperfine.pt", num_time_steps=1000, learning_rate=3e-5, dropout=0.0, load_file="PTFiles/ema_deeperef.pt")
    train_unet(epochs=60, batch_size=16, file_base="ema_deeperfine2.pt", num_time_steps=1000, learning_rate=1e-5, dropout=0.0, load_file="PTFiles/ema_deeperfine.pt")
    train_unet(epochs=60, batch_size=8, file_base="ema_deeperfine3.pt", num_time_steps=1000, learning_rate=5e-6, dropout=0.0, load_file="PTFiles/ema_deeperfine2.pt")
    # train_unet(epochs=200, batch_size=64, file_base="unconditionalref.pt", num_time_steps=1000, learning_rate=5e-5, dropout=0.0, load_file="PTFiles/unconditional.pt")
    # train_unet(epochs=50, batch_size=256, file_base="refined.pt", num_time_steps=1000, learning_rate=5e-7, dropout=0.0, load_file="PTFiles/thousand.pt")
    # train_unet(epochs=200, batch_size=256, file_base="thousand.pt", num_time_steps=100, learning_rate=1e-4)