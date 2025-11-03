import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import mushroomdata
import NetworkComponents as nc

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        #following resnet up to 512:
        # 1: getting intput into 64x64x64 by doing a 3x64x64 -> 64x64x64 transformation
        self.in_to_conv1 = nn.Conv2d(3, 32, kernel_size=(1, 1), padding=1)  # 3x64x64 -> 32x64x64
        self.norm_conv1 = nn.BatchNorm2d(32)
        self.bn_block1 = nc.BottleneckBlock(32, 8)  # 32x64x64 -> 32x64x64
        # self.increase_conv1_channels = nn.Conv2d(32, 64, kernel_size=(1, 1)) # 32x64x64 -> 64x64x64
        # self.norm_conv1_2 = nn.BatchNorm2d(64)
        self.conv1_to_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 32x64x64 → 64x32x32
        self.norm_conv2 = nn.BatchNorm2d(64)
        self.bn_block2 = nc.BottleneckBlock(64, 16) #64x32x32 -> 64x32x32
        self.conv2_to_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 64x32x32 → 128x16x16
        self.norm_conv3 = nn.BatchNorm2d(128)
        self.bn_block3 = nc.BottleneckBlock(128, 32) #128x16x16 -> #128x16x16
        self.conv3_to_conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) #128x16x16 → 256x8x8
        self.norm_conv4 = nn.BatchNorm2d(256)
        self.bn_block4 = nc.BottleneckBlock(256, 64) #256x8x8 → 256x8x8

        self.conv4_to_mean = nn.Conv2d(256, 4, kernel_size=1)  # 256x8x8 -> 4x8x8
        self.conv4_to_log_var = nn.Conv2d(256, 4, kernel_size=1)  # 256x8x8 -> 4x8x8

    def forward(self, x):
        x = F.leaky_relu(self.norm_conv1(self.in_to_conv1(x)))
        x = self.bn_block1(x)
        # x = F.leaky_relu(self.norm_conv1_2(self.increase_conv1_channels(x)))
        x = F.leaky_relu(self.norm_conv2(self.conv1_to_conv2(x)))
        x = self.bn_block2(x)
        x = F.leaky_relu(self.norm_conv3(self.conv2_to_conv3(x)))
        x = self.bn_block3(x)
        x = F.leaky_relu(self.norm_conv4(self.conv3_to_conv4(x)))
        x = self.bn_block4(x)

        mean = self.conv4_to_mean(x)
        log_var = self.conv4_to_log_var(x)
        log_var = torch.clamp(log_var, -10, 10)
        std = torch.exp(0.5 * log_var)
        KL_div = -0.5 * (1 + log_var - mean ** 2.0 - log_var.exp()).sum(dim=[1, 2, 3]).mean(dim=0)  # KL-Div
        return mean + std * torch.randn_like(log_var), KL_div

    def forward_static(self, x):
        x = F.leaky_relu(self.norm_conv1(self.in_to_conv1(x)))
        x = self.bn_block1(x)
        # x = F.leaky_relu(self.norm_conv1_2(self.increase_conv1_channels(x)))
        x = F.leaky_relu(self.norm_conv2(self.conv1_to_conv2(x)))
        x = self.bn_block2(x)
        x = F.leaky_relu(self.norm_conv3(self.conv2_to_conv3(x)))
        x = self.bn_block3(x)
        x = F.leaky_relu(self.norm_conv4(self.conv3_to_conv4(x)))
        x = self.bn_block4(x)

        mean = self.conv4_to_mean(x)
        return mean

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_to_conv1 = nn.ConvTranspose2d(4, 16, kernel_size=3, stride=2, padding=1)  # 4x8x8 -> 16x16x16
        self.bn_block1 = nc.BottleneckBlock(16, 4)
        self.conv1_to_conv2 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1)  # 16x16x16 -> 32x32x32
        self.bn_block2 = nc.BottleneckBlock(32, 8)
        self.conv2_to_conv3 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2, padding=1)  # 32x32x32 -> 64x64x64
        self.bn_block3 = nc.BottleneckBlock(64, 16)
        self.conv3_to_out = nn.Conv2d(64, 3, kernel_size=1)  # 64x64x64 -> 3x64x64

    def forward(self, x):
        x = F.leaky_relu(self.latent_to_conv1(x))
        x = self.bn_block1(x)
        x = F.leaky_relu(self.conv1_to_conv2(x))
        x = self.bn_block2(x)
        x = F.leaky_relu(self.conv2_to_conv3(x))
        x = self.bn_block3(x)

        return self.conv3_to_out(x)

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x, KL_div = self.encoder(x)
        return self.decoder(x), KL_div


def train_nn(epochs=15, batch_size=32, lr=0.001, num_periods=5):
    #Load the picture data
    dataset = mushroomdata.MushroomData("DataJsons/traindirs.json")
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}.")

    model = AutoEncoder().to(device)

    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    beta_multiplier = 1.0
    period_length = epochs * len(dataloader) / num_periods
    batch_idx = 0

    for epoch in range(epochs):
        p_bar = tqdm(dataloader, desc=f"Epoch [{epoch + 1} / {epochs}]")
        for images in p_bar:
            images = images[0].to(device)

            beta = beta_multiplier * np.sin(np.pi * (batch_idx % period_length) / period_length)

            optimizer.zero_grad()
            outputs, KL_div = model(images)
            loss = loss_fn(outputs, images) + beta * KL_div
            loss.backward()
            optimizer.step()

            batch_idx += 1
            p_bar.set_postfix({'Loss': loss.item(), 'KL_div': KL_div.item(), 'beta': beta})

    torch.save(model.state_dict(), "mushroom_vae.pt")

epochs = 5
batch_size = 64
train_nn(epochs, batch_size)