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
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch_ema import ExponentialMovingAverage

class UNET(nn.Module):

    def __init__(self, time_embed_dim, label_embed_dim, num_classes, dropout_p=0.0, starting_scale=8):
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
        self.label_embed = nn.Embedding(num_classes, label_embed_dim)
        self.label_mlp = nn.Sequential(
            nn.Linear(label_embed_dim, label_embed_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(label_embed_dim*4, label_embed_dim),
            nn.SiLU()
        )

        # Downward pass of the UNet
        self.initial = nn.Conv2d(8, 48, 1) # 8 x 8 x 8 -> 32 x 8 x 8

        self.pass1 = nc.UNetLayer(48, starting_scale, time_embed_dim, label_embed_dim, dropout_p) # 32 x 8 x 8 retained throughout
        self.down1 = nn.Conv2d(48, 96, 3, 2, 1) # 32 x 8 x 8 -> 64 x 4 x 4

        self.pass2 = nc.UNetLayer(96, starting_scale//2, time_embed_dim, label_embed_dim, dropout_p) # 64 x 4 x 4
        self.down2 = nn.Conv2d(96, 192, 3, 2, 1) # 64 x 4 x 4 -> 128 x 2 x 2

        self.pass3 = nc.UNetLayer(192, starting_scale//4, time_embed_dim, label_embed_dim, dropout_p) # 128 x 2 x 2
        self.up1 = nn.ConvTranspose2d(192, 96, 2, 2) # 128 x 2 x 2 -> 64 x 4 x 4
        # Concatenation here -> 128 x 4 x 4

        self.pass4 = nc.UNetLayer(192, starting_scale//2, time_embed_dim, label_embed_dim, dropout_p) # 128 x 4 x 4
        self.up2 = nn.ConvTranspose2d(192, 48, 2, 2) # 128 x 4 x 4 -> 32 x 8 x 8
        # Concatenation here -> 64 x 8 x 8

        # self.pass5 = nc.NResBlocks(2, 64, 32, 8, time_embed_dim, label_embed_dim, dropout_p)
        self.pass5 = nc.UNetLayer(96, starting_scale, time_embed_dim, label_embed_dim, dropout_p)
        self.to_out = nn.Conv2d(96, 8, 1)



    def forward(self, x, time_embed, l):
        global_t = self.time_mlp(time_embed)
        global_l = self.label_mlp(self.label_embed(l))

        x = self.initial(x)

        res1 = self.pass1(x, global_t, global_l) # 32 x 8 x 8

        res2 = self.down1(res1)
        res2 = self.pass2(res2, global_t, global_l) # 64 x 4 x 4

        res3 = self.down2(res2)
        res3 = self.pass3(res3, global_t, global_l) # 128 x 2 x 2

        up = self.up1(res3) # 64 x 4 x 4
        up = torch.cat([up, res2], dim=1) # 128 x 4 x 4
        up = self.pass4(up, global_t, global_l)

        up = self.up2(up) # 32 x 8 x 8
        up = torch.cat([up, res1], dim=1) # 64 x 8 x 8
        up = self.pass5(up, global_t, global_l)

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
from torch.utils.data import WeightedRandomSampler

def train_unet(epochs=15, batch_size = 32, learning_rate = 0.001, num_time_steps = 1000, file_base = "unet.pt",
               vae_file = "PTFiles/largernorm3.pt", vae_latent_channels=8, dropout=0.0, load_file=None, previous_epochs=0,
               warmup_steps=2500, latent_width = 8, given_vae=None, num_classes=100):
    # dataset = mushroomdata.MushroomData("DataJsons/traindirs.json")
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
    # beta_steps = np.array([start_step + (end_step - start_step)*i/(num_time_steps-1) for i in range(num_time_steps)])

    betas = torch.linspace(start_step, end_step, num_time_steps, device=device)
    alphas = 1 - betas

    alpha_bars = torch.zeros(num_time_steps, device=device)
    alpha_bars[0] = alphas[0]
    for i in range(1, num_time_steps):
        alpha_bars[i] = alphas[i] * alpha_bars[i-1]
    
    # betas, alphas, alpha_bars = cosine_beta_schedule(num_time_steps)
    # alpha_bars = alpha_bars.to(device)

    unet_model = UNET(128, 256, num_classes, dropout_p=dropout, starting_scale=16).to(device)
    ema = ExponentialMovingAverage(unet_model.parameters(), decay=0.9999)
    ema.to(device)

    if load_file is not None:
        checkpoint = torch.load(load_file, map_location=device)
        unet_model.load_state_dict(checkpoint['model'])
        ema.load_state_dict(checkpoint['ema'])
        # unet_model.load_state_dict(torch.load(load_file, map_location=device))
    # unet_model = unet_model.to(device=device)

    time_encodings = nc.positional_encoding(num_time_steps, 128).to(device=device) # (num_time_steps, 64) array of time encodings

    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=learning_rate)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=epochs * (len(dataset) / batch_size),   
    #     eta_min=5e-6
    # )

    total_steps = epochs * len(dataset) / batch_size
    min_lr_ratio = 0.02
    already_done_steps = previous_epochs * len(dataset) / batch_size

    def lr_lambda(step):
        step = step + already_done_steps
        # ----------- Warmup -------------
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        # ----------- Cosine Decay --------
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))

        # final LR is min_lr_ratio * base_lr
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
            # used_alpha_bars = torch.tensor(alpha_bars, dtype=torch.float32, device=device)[time_steps].view(-1, 1, 1, 1)
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
        if (epoch + 1) % 40 == 0:
            torch.save({'model' : unet_model.state_dict(), 'ema' : ema.state_dict()}, f"PTFiles/inprogress{epoch}{file_base}")

if __name__ == '__main__':
    pass

    # batch size could be too big (256 -> 64 -> 32?)
    # train on just the mu, not mu + std (Done!)
    # In diffusion process - sample from distribution generated by means (latent vectors)
    # Can select another scaling factor ~0.4 multiplied by the latent standard deviation
    # Gets closer convergence to the mean\

    # unet_model = UNET(64, 128, 100)
    # torch.save({'model': unet_model.state_dict()}, 'savetest.pt')
    # unet_model = UNET(64, 128, 100)
    # unet_model.load_state_dict(torch.load("savetest.pt")['model'])
    # print('hi')

    # train_unet(epochs=50, batch_size=64, file_base="attention.pt", num_time_steps=1000, learning_rate=1e-4, dropout=0.1)
    # train_unet(epochs=50, batch_size=64, file_base="attention1.pt", num_time_steps=1000, learning_rate=1e-4, dropout=0.1, load_file="PTFiles/attention.pt")
    # train_unet(epochs=50, batch_size=64, file_base="attention2.pt", num_time_steps=1000, learning_rate=5e-5, dropout=0.1, load_file="PTFiles/attention1.pt")

    # train_unet(epochs=150, batch_size=64, file_base="attention3.pt", num_time_steps=1000, learning_rate=5e-5, dropout=0.1, load_file="PTFiles/attention2.pt")
    # train_unet(epochs=250, batch_size=64, file_base="deeper_atten2.pt", num_time_steps=1000, learning_rate=1e-4, dropout=0.1, load_file="PTFiles/deeper_atten2.pt", previous_epochs=190)
    # train_unet(epochs=150, batch_size=64, file_base="deeper_atten3.pt", num_time_steps=1000, learning_rate=3e-5, dropout=0.1, load_file="PTFiles/deeper_atten2.pt", previous_epochs=0, warmup_steps=0)

    # 150 - lr around 3e-5
    # 190

    # train_unet(epochs=200, batch_size=64, file_base="more_channels.pt", num_time_steps=1000, learning_rate=1e-4, dropout=0, previous_epochs=110, load_file="PTFiles/more_channels.pt")

    train_unet(epochs=200, batch_size=64, file_base="new_decoder_unet2.pt", num_time_steps=1000, learning_rate=1e-4,
               dropout=0, previous_epochs=160, vae_file="PTFiles/attn_vae_64x64.pt", latent_width=16, load_file="PTFiles/new_decoder_unet.pt")
    train_unet(epochs=200, batch_size=64, file_base="new_decoder_unetref.pt", num_time_steps=1000, learning_rate=3e-5,
               dropout=0, previous_epochs=52, vae_file="PTFiles/attn_vae_64x64.pt", latent_width=16, load_file="PTFiles/new_decoder_unet.pt", warmup_steps=0)
    # 124
    # train_unet(epochs=100, batch_size=64, file_base="reworkedref.pt", num_time_steps=1000, learning_rate=3.5e-5, dropout=0, load_file="PTFiles/reworked.pt", warmup_steps=0)

    # train_unet(epochs=50, batch_size=32, file_base="smol_attention.pt", num_time_steps=1000, learning_rate=1e-4, dropout=0.1)
    # train_unet(epochs=50, batch_size=32, file_base="smol_attention1.pt", num_time_steps=1000, learning_rate=1e-4, dropout=0.1, load_file="PTFiles/smol_attention.pt")
    # train_unet(epochs=50, batch_size=32, file_base="smol_attention2.pt", num_time_steps=1000, learning_rate=5e-5, dropout=0.1, load_file="PTFiles/smol_attention1.pt")

    # train_unet(epochs=40, batch_size=64, file_base="conditional_ema2_1.pt", num_time_steps=1000, learning_rate=1e-4, dropout=0.0)
    # train_unet(epochs=60, batch_size=32, file_base="conditional_ema2_2.pt", num_time_steps=1000, learning_rate=1e-4, dropout=0.0, load_file="PTFiles/conditional_ema2_1.pt")
    # train_unet(epochs=80, batch_size=16, file_base="conditional_ema2_3.pt", num_time_steps=1000, learning_rate=5e-5, dropout=0.0, load_file="PTFiles/conditional_ema2_3.pt")
    # train_unet(epochs=80, batch_size=8, file_base="conditional_ema2_4.pt", num_time_steps=1000, learning_rate=2e-5, dropout=0.0, load_file="PTFiles/conditional_ema2_3.pt")

    # train_unet(epochs=200, batch_size=64, file_base="ema_deeper.pt", num_time_steps=1000, learning_rate=1e-4, dropout=0.0)
    # train_unet(epochs=100, batch_size=64, file_base="ema_deeperef.pt", num_time_steps=1000, learning_rate=5e-5, dropout=0.0, load_file="PTFiles/ema_deeper.pt")
    # train_unet(epochs=60, batch_size=32, file_base="ema_deeperfine.pt", num_time_steps=1000, learning_rate=3e-5, dropout=0.0, load_file="PTFiles/ema_deeperef.pt")
    # train_unet(epochs=60, batch_size=16, file_base="ema_deeperfine2.pt", num_time_steps=1000, learning_rate=1e-5, dropout=0.0, load_file="PTFiles/ema_deeperfine.pt")
    # train_unet(epochs=60, batch_size=8, file_base="ema_deeperfine3.pt", num_time_steps=1000, learning_rate=5e-6, dropout=0.0, load_file="PTFiles/ema_deeperfine2.pt")
    # train_unet(epochs=200, batch_size=64, file_base="unconditionalref.pt", num_time_steps=1000, learning_rate=5e-5, dropout=0.0, load_file="PTFiles/unconditional.pt")
    # train_unet(epochs=50, batch_size=256, file_base="refined.pt", num_time_steps=1000, learning_rate=5e-7, dropout=0.0, load_file="PTFiles/thousand.pt")
    # train_unet(epochs=200, batch_size=256, file_base="thousand.pt", num_time_steps=100, learning_rate=1e-4)