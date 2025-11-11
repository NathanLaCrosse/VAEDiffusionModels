import cv2
import numpy as np
import torch
from fontTools.unicodedata import block
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import dropout
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plot

import NetworkComponents as nc
import mushroomdata


class NBottleneckBlocks(nn.Module):

    def __init__(self, n, initial_channels, bottleneck_channels, dropout_p=0.0):
        super(NBottleneckBlocks, self).__init__()

        self.net = nn.ModuleList(
            [nc.BottleneckBlock(initial_channels, bottleneck_channels, dropout_p) for i in range(n)]
        )

    def forward(self, x):
        for i in range(len(self.net)):
            x = self.net[i](x)
        return x



class Encoder(nn.Module):
    """
    Compress an image down to a latent space of 4 x 8 x 8
    Utilizes residual layers similar to a ResNet model
    """
    def __init__(self, latent_channels=4, dropout=0.0):
        super(Encoder, self).__init__()

        self.initial = nn.Conv2d(3, 16, 3, 1, 1) # 16 x 64 x 64
        self.layer1 = NBottleneckBlocks(5, 16, 4, dropout_p=dropout) # 16 x 64 x 64
        self.down1 = nn.Conv2d(16, 32, 3, 2, 1) # 32 x 32 x 32
        self.layer2 = NBottleneckBlocks(5, 32, 8, dropout_p=dropout) # 32 x 32 x 32
        self.down2 = nn.Conv2d(32, 64, 3, 2, 1) # 64 x 16 x 16
        self.layer3 = NBottleneckBlocks(5, 64, 16, dropout_p=dropout) # 64 x 16 x 16
        self.down3 = nn.Conv2d(64, 128, 3, 2, 1) # 128 x 8 x 8
        self.layer4 = NBottleneckBlocks(5, 128, 32, dropout_p=dropout) # 128 x 8 x 8

        # Extract mean and logvar -> 4 x 8 x 8
        self.to_mean = nn.Conv2d(128, latent_channels, 1)
        self.to_logvar = nn.Conv2d(128, latent_channels, 1)

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.down1(x)
        x = self.layer2(x)
        x = self.down2(x)
        x = self.layer3(x)
        x = self.down3(x)
        x = self.layer4(x)

        mean = self.to_mean(x)
        logvar = self.to_logvar(x)

        logvar = torch.clamp(logvar, -10, 10) # Prevent overflow in std calculation
        std = torch.exp(0.5 * logvar)
        KL_div = -0.5 * (1 + logvar - mean ** 2.0 - logvar.exp()).sum(dim=[1, 2, 3]).mean(dim=0) # Loss term

        return mean + std * torch.randn_like(logvar), KL_div

    def forward_to_mean_std(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.down1(x)
        x = self.layer2(x)
        x = self.down2(x)
        x = self.layer3(x)
        x = self.down3(x)
        x = self.layer4(x)

        mean = self.to_mean(x)
        logvar = self.to_logvar(x)

        logvar = torch.clamp(logvar, -10, 10)  # Prevent overflow in std calculation
        std = torch.exp(0.5 * logvar)

        return mean, std

    def forward_static(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.down1(x)
        x = self.layer2(x)
        x = self.down2(x)
        x = self.layer3(x)
        x = self.down3(x)
        x = self.layer4(x)

        mean = self.to_mean(x)
        return mean


class Decoder(nn.Module):
    """
    Converts a latent back into an image.
    """
    def __init__(self, latent_channels=4, dropout=0.0):
        super(Decoder, self).__init__()

        self.initial = nn.Conv2d(latent_channels, 128, 3, 1, 1) # 128 x 8 x 8
        self.layer1 = NBottleneckBlocks(5, 128, 32, dropout_p=dropout) # 128 x 8 x 8
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2) # 64 x 16 x 16
        self.layer2 = NBottleneckBlocks(5, 64, 16, dropout_p=dropout) # 64 x 16 x 16
        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2) # 32 x 32 x 32
        self.layer3 = NBottleneckBlocks(5, 32, 8, dropout_p=dropout) # 32 x 32 x 32
        self.up3 = nn.ConvTranspose2d(32, 16, 2, 2) # 16 x 64 x 64
        self.layer4 = NBottleneckBlocks(5, 16, 4, dropout_p=dropout) # 16 x 64 x 64
        self.out = nn.Conv2d(16, 3, 3, 1, 1) # 3 x 64 x 64

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.up1(x)
        x = self.layer2(x)
        x = self.up2(x)
        x = self.layer3(x)
        x = self.up3(x)
        x = self.layer4(x)
        return F.tanh(self.out(x)) # force output to be between -1 & 1


class VAE(nn.Module):
    def __init__(self, latent_channels=4, dropout=0.0):
        super(VAE, self).__init__()

        self.encoder = Encoder(latent_channels=latent_channels, dropout=dropout)
        self.decoder = Decoder(latent_channels=latent_channels, dropout=dropout)

    def forward(self, x):
        x, KL_div = self.encoder(x)
        return self.decoder(x), KL_div

    def forward_to_mean_std(self, x):
        return self.encoder.forward_to_mean_std(x)

    def forward_decode_only(self, x):
        return self.decoder(x)

    def forward_encode_only(self, x):
        x, _ = self.encoder(x)
        return x

def train_nn(epochs=15, batch_size=32, lr=0.001, num_periods=5):
    #Load the picture data
    dataset = mushroomdata.MushroomData("DataJsons/traindirs.json")
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}.")

    model = VAE().to(device)

    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    beta_multiplier = 1.0
    period_length = epochs * len(dataloader) / num_periods
    batch_idx = 0

    cv2.namedWindow("Original / Reconstruction")

    losses = []
    for epoch in range(epochs):
        p_bar = tqdm(dataloader, desc=f"Epoch [{epoch + 1} / {epochs}]")
        for images in p_bar:
            images = images[0].to(device)

            beta = beta_multiplier * np.sin(np.pi * (batch_idx % period_length) / period_length)

            optimizer.zero_grad()
            outputs, KL_div = model(images)
            loss = loss_fn(outputs, images) + beta * KL_div
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            batch_idx += 1
            p_bar.set_postfix({'Loss': loss.item(), 'KL_div': KL_div.item(), 'beta': beta})

            if batch_idx % 4 == 0:
                with torch.no_grad():
                    img = images[0].unsqueeze(0)
                    latent = model.encoder.forward_static(img)
                    reconstructed_img = F.sigmoid(model.decoder(latent)[0]).cpu().numpy()
                img = np.concatenate((img[0].cpu().numpy()[0], reconstructed_img[0]), axis=1)
                cv2.imshow("Original / Reconstruction", cv2.resize(np.uint8(255 * img), (560, 280)))
                cv2.waitKey(1)
                # if len(losses) < 200:
                plot.plot(losses)
                # else:
                #     plot.close()
                #     plot.plot(losses[-199:])
                plot.show(block=False)
                plot.pause(0.001)
        cv2.destroyAllWindows()

    torch.save(model.state_dict(), "mushroom_vae.pt")

epochs = 5
batch_size = 64
train_nn(epochs, batch_size)