import torch
from torch import nn
import torch.nn.functional as F
import NetworkComponents as nc

class Encoder(nn.Module):
    def __init__(self, latent_channels=8, dropout=0.0):
        super(Encoder, self).__init__()
        self.initial = nn.Conv2d(3, 16, 3, 1, 1) # 16 x 64 x 64
        self.res1 = nc.NVAEResBlocks(2, 16, 16)
        self.attn1 = nc.SelfAttention(16)
        self.down1 = nn.Conv2d(16, 32, 3, 2, 1) # 32 x 32 x 32
        self.res2 = nc.NVAEResBlocks(2, 32, 32)
        self.attn2 = nc.SelfAttention(32)
        self.down2 = nn.Conv2d(32, 64, 3, 2, 1) # 64 x 16 x 16
        self.res3 = nc.NVAEResBlocks(2, 64, 64)
        self.attn3 = nc.SelfAttention(64)
        self.down3 = nn.Conv2d(64, 128, 3, 2, 1) # 128 x 8 x 8
        self.res4 = nc.NVAEResBlocks(2, 128, 128)
        self.attn4 = nc.SelfAttention(128)

        self.to_mean = nn.Conv2d(128, latent_channels, 1)
        self.to_logvar = nn.Conv2d(128, latent_channels, 1)

    def forward(self, x):
        x = self.initial(x)
        x = self.res1(x)
        # x = self.attn1(x)
        x = self.down1(x)
        x = self.res2(x)
        # x = self.attn2(x)
        x = self.down2(x)
        x = self.res3(x)
        x = self.attn3(x)
        x = self.down3(x)
        x = self.res4(x)
        x = self.attn4(x)

        mean = self.to_mean(x)
        logvar = self.to_logvar(x)

        logvar = torch.clamp(logvar, -10, 10) # Prevent overflow in std calculation
        std = torch.exp(0.5 * logvar)
        KL_div = -0.5 * (1 + logvar - mean ** 2.0 - logvar.exp()).sum(dim=[1, 2, 3]).mean(dim=0) # Loss term

        return mean + std * torch.randn_like(logvar), KL_div

class Decoder(nn.Module):
    """
    Converts a latent back into an image.
    """
    def __init__(self, latent_channels=8, dropout=0.0):
        super(Decoder, self).__init__()

        self.initial = nn.Conv2d(latent_channels, 128, 3, 1, 1) # 128 x 8 x 8
        self.res1 = nc.NVAEResBlocks(2, 128, 128)
        self.attn1 = nc.SelfAttention(128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2) # 64 x 16 x 16
        self.res2 = nc.NVAEResBlocks(2, 64, 64) # 64 x 16 x 16
        self.attn2 = nc.SelfAttention(64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2) # 32 x 32 x 32
        self.res3 = nc.NVAEResBlocks(2, 32, 32) # 32 x 32 x 32
        self.up3 = nn.ConvTranspose2d(32, 16, 2, 2) # 16 x 64 x 64
        self.res4 = nc.NVAEResBlocks(2, 16, 16) # 16 x 64 x 64
        self.out = nn.Conv2d(16, 3, 3, 1, 1) # 3 x 64 x 64

    def forward(self, x):
        x = self.initial(x)
        x = self.res1(x)
        x = self.attn1(x)
        x = self.up1(x)
        x = self.res2(x)
        x = self.attn2(x)
        x = self.up2(x)
        x = self.res3(x)
        x = self.up3(x)
        x = self.res4(x)
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