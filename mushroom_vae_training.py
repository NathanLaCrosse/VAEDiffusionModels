import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import mushroomdata
from NathanVAE import VAE
import torchvision.models as models
import lpips

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        #Note:: Can do kernal_size(9, 9) to shrink by 8 each convolution, or do step=2, kernal_size=(5, 5)
        self.in_to_conv1 = nn.Conv2d(3, 32, kernel_size=(9, 9))  # 3x64x64 -> 32x56x56
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.conv1_to_conv2 = nn.Conv2d(32, 64, kernel_size=(9,9))  # 32x56x56 -> 64x48x48
        self.bn_conv2 = nn.BatchNorm2d(64)
        self.conv2_to_conv3 = nn.Conv2d(64, 128, kernel_size=(9, 9))  # 64x48x48 -> 128x40x40
        self.bn_conv3 = nn.BatchNorm2d(128)
        self.conv3_to_conv4 = nn.Conv2d(128, 256, kernel_size=(9, 9))  # 128x40x40 -> 256x32x32
        self.bn_conv4 = nn.BatchNorm2d(256)
        self.conv4_to_conv5 = nn.Conv2d(256, 512, kernel_size=(9, 9))  # 256x32x32 -> 512x24x24
        self.bn_conv5 = nn.BatchNorm2d(512)
        self.conv5_to_conv6 = nn.Conv2d(512, 1024, kernel_size=(9, 9))  # 512x24x24 -> 1024x16x16
        self.bn_conv6 = nn.BatchNorm2d(1024)
        self.conv6_to_mean = nn.Conv2d(1024, 4, kernel_size=(9, 9))  # 1024x16x16 -> 4x8x8
        self.conv6_to_log_var = nn.Conv2d(1024, 4, kernel_size=(9, 9))  # 1024x16x16 -> 4x8x8

    def forward(self, x):
        x = F.leaky_relu(self.bn_conv1(self.in_to_conv1(x)))
        x = F.leaky_relu(self.bn_conv2(self.conv1_to_conv2(x)))
        x = F.leaky_relu(self.bn_conv3(self.conv2_to_conv3(x)))
        x = F.leaky_relu(self.bn_conv4(self.conv3_to_conv4(x)))
        x = F.leaky_relu(self.bn_conv5(self.conv4_to_conv5(x)))
        x = F.leaky_relu(self.bn_conv6(self.conv5_to_conv6(x)))
        mean = self.conv6_to_mean(x)
        log_var = self.conv6_to_log_var(x)
        log_var = torch.clamp(log_var, -10, 10)
        std = torch.exp(0.5 * log_var)
        KL_div = -0.5 * (1 + log_var - mean ** 2.0 - log_var.exp()).sum(dim=[1, 2, 3]).mean(dim=0)  # KL-Div
        return mean + std * torch.randn_like(log_var), KL_div

    def forward_static(self, x):
        x = F.leaky_relu(self.bn_conv1(self.in_to_conv1(x)))
        x = F.leaky_relu(self.bn_conv2(self.conv1_to_conv2(x)))
        x = F.leaky_relu(self.bn_conv3(self.conv2_to_conv3(x)))
        x = F.leaky_relu(self.bn_conv4(self.conv3_to_conv4(x)))
        x = F.leaky_relu(self.bn_conv5(self.conv4_to_conv5(x)))
        x = F.leaky_relu(self.bn_conv6(self.conv5_to_conv6(x)))
        mean = self.conv6_to_mean(x)
        return mean

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_to_conv1 = nn.ConvTranspose2d(4, 32, kernel_size=(9, 9))  # 4x8x8 -> 32x16x16
        self.conv1_to_conv2 = nn.ConvTranspose2d(32, 64, kernel_size=(9, 9))  # 32x16x16 -> 64x24x24
        self.conv2_to_conv3 = nn.ConvTranspose2d(64, 128, kernel_size=(9, 9))  # 64x24x24 -> 128x32x32
        self.conv3_to_conv4 = nn.ConvTranspose2d(128, 256, kernel_size=(9, 9))  # 128x32x32 -> 256x40x40
        self.conv4_to_conv5 = nn.ConvTranspose2d(256, 512, kernel_size=(9, 9))  # 256x40x40 -> 512x48x48
        self.conv5_to_conv6 = nn.ConvTranspose2d(512, 1024, kernel_size=(9,9))
        self.conv6_to_out = nn.ConvTranspose2d(1024, 3, kernel_size=(9, 9))  # 512x48x48 -> 3x64x64

    def forward(self, x):
        x = F.leaky_relu(self.latent_to_conv1(x))
        x = F.leaky_relu(self.conv1_to_conv2(x))
        x = F.leaky_relu(self.conv2_to_conv3(x))
        x = F.leaky_relu(self.conv3_to_conv4(x))
        x = F.leaky_relu(self.conv4_to_conv5(x))
        x = F.leaky_relu(self.conv5_to_conv6(x))
        return self.conv6_to_out(x)

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x, KL_div = self.encoder(x)
        return self.decoder(x), KL_div





# class PerceptualLoss(nn.Module):
#     """
#     Loads a pre-trained model (vgg19) to compute perceptual loss.
#     """
#     def __init__(self, layers=[8, 17, 26]):
#         super().__init__()
#         vgg = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1).features.eval()
#         self.vgg_layers = vgg
#         self.layers = layers
#         for param in self.vgg_layers.parameters():
#             param.requires_grad = False
#
#     def forward(self, pred, target):
#         loss = torch.zeros(1)
#         for i, layer in enumerate(self.vgg_layers):
#             pred = layer(pred)
#             target = layer(target)
#             if i in self.layers:
#                 loss += torch.nn.functional.l1_loss(pred, target, reduction="sum")
#         return loss


def train_nn(epochs=15, batch_size=32, lr=0.001, num_periods=5, beta_mult=0.1, percep_mult=1, save_file="mushroom_vae.pt", load_file=None, latent_channels=4, mse_mode=False):
    #Load the picture data
    dataset = mushroomdata.MushroomData("DataJsons/traindirs.json", mse_mode=mse_mode)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}.")

    model = VAE(latent_channels=latent_channels).to(device)

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

        torch.save(model.state_dict(), f"PTFiles/{save_file}")
        # if (epoch+1) % 30 == 0:
        #     torch.save(model.state_dict(), f"PTFiles/inprogress{epoch}{save_file}")

epochs = 100
batch_size = 256
train_nn(epochs, batch_size, lr=0.001, num_periods=5, beta_mult=0.0002, percep_mult=0.02, save_file="tinypercep1.pt", load_file=None, latent_channels=8, mse_mode=True)
train_nn(epochs=20, batch_size=batch_size, lr=0.001, num_periods=0.75, beta_mult=0.0002, percep_mult=0.02, save_file="tinypercep2.pt", load_file="PTFiles/tinypercep1.pt", latent_channels=8, mse_mode=True)
train_nn(epochs=20, batch_size=batch_size, lr=0.001, num_periods=0.5, beta_mult=0.0002, percep_mult=0.02, save_file="tinypercep3.pt", load_file="PTFiles/tinypercep1.pt", latent_channels=8, mse_mode=True)
# train_nn(epochs, batch_size, lr=0.001, num_periods=0.5, beta_mult=0.01, percep_mult=0.015, save_file="percep1.pt", load_file=None, latent_channels=16, mse_mode=True)
# train_nn(epochs, batch_size, lr=0.001, num_periods=0.5, beta_mult=0.001, percep_mult=0.015, save_file="percep2.pt", load_file=None, latent_channels=16, mse_mode=True)
# train_nn(epochs, batch_size, lr=0.001, num_periods=0.75, beta_mult=0.0001, percep_mult=0.015, save_file="smaller_percep.pt", load_file=None, latent_channels=10, mse_mode=True)
# train_nn(epochs, batch_size, lr=0.001, num_periods=0.5, beta_mult=0.00001, percep_mult=0.015, save_file="percep4.pt", load_file=None, latent_channels=16, mse_mode=True)
# train_nn(epochs, batch_size, lr=0.001, num_periods=0.5, beta_mult=0.000001, percep_mult=0.015, save_file="percep5.pt", load_file=None, latent_channels=16, mse_mode=True)
# train_nn(epochs, batch_size, lr=0.001, num_periods=25.5, beta_mult=15, percep_mult=2e6, save_file="percep2.pt", load_file=None, latent_channels=16)
# train_nn(epochs, batch_size, lr=0.001, num_periods=13.75, beta_mult=15, percep_mult=2e6, save_file="percep3.pt", load_file=None, latent_channels=16)
# train_nn(epochs, batch_size, lr=0.001, num_periods=10, beta_mult=50, save_file="beta1.pt", load_file=None, latent_channels=16)
# train_nn(epochs, batch_size, lr=0.001, num_periods=10, beta_mult=100, save_file="beta2.pt", load_file=None, latent_channels=16)
# train_nn(epochs, batch_size, lr=0.001, num_periods=10, beta_mult=500, save_file="beta3.pt", load_file=None, latent_channels=16)
# train_nn(epochs, batch_size, lr=0.001, num_periods=10, beta_mult=1000, save_file="beta4.pt", load_file=None, latent_channels=16)
# train_nn(epochs, batch_size, lr=0.001, num_periods=10, beta_mult=0.1, save_file="mushroom_vae2.pt")
# train_nn(epochs, batch_size, lr=0.001, num_periods=10, beta_mult=1, save_file="mushroom_vae3.pt")
# train_nn(epochs, batch_size, lr=0.001, num_periods=10, beta_mult=10, save_file="mushroom_vae4.pt")
# train_nn(epochs, batch_size, lr=0.001, num_periods=10, beta_mult=100, save_file="mushroom_vae5.pt")

# Percep loss starts around 0.4 - check on later
# beta 0.01 too much