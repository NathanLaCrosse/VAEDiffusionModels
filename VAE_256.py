import torch
from torch import nn
import torch.nn.functional as F
import NetworkComponents as nc
import AttentionComponents as ac

class Encoder(nn.Module):
    def __init__(self, latent_channels=4, dropout=0.0):
        super(Encoder, self).__init__()
        self.initial = nn.Conv2d(3, 16, 3, 1, 1) # 16 x 256 x 256

        self.res1 = nc.NVAEResBlocks(2, 16, 16)
        self.down1 = nc.VAEResnetBlock(16, 32, True)  # 32 x 128 x 128

        self.res2 = nc.NVAEResBlocks(3, 32, 32)
        self.down2 = nc.VAEResnetBlock(32, 64, True) # 64x64x64

        self.res3 = nc.NVAEResBlocks(3, 64, 64)
        self.down3 = nc.VAEResnetBlock(64, 128, True) # 128x32x32

        self.res4 = nc.NVAEResBlocks(2, 128, 128)

        #Attention is not traditionally used, but could be beneficial
        self.attnEnc = ac.MultiHeadSelfAttention(128, 8)

        self.to_mean = nn.Conv2d(128, latent_channels, 1)
        self.to_logvar = nn.Conv2d(128, latent_channels, 1)

    def forward(self, x):
        x = self.initial(x)
        x = self.res1(x)
        x = self.down1(x)
        x = self.res2(x)
        x = self.down2(x)
        x = self.res3(x)
        x = self.down3(x)
        x = self.res4(x)
        # x = self.attnEnc(x)
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
    def __init__(self, latent_channels=4, dropout=0.0):
        super(Decoder, self).__init__()

        self.initial = nn.Conv2d(latent_channels, 128, 1) # 256 x 32 x 32

        #Attention is here as well just in case
        self.attnDec = ac.MultiHeadSelfAttention(128, 8)

        self.res1 = nc.NVAEResBlocks(2, 128, 128) # 128 x 32 x 32
        self.up1 = nc.VAEResnetBlock(128, 64, False) # 64 x 64 x 64

        self.res2 = nc.NVAEResBlocks(3, 64, 64) # 64 x 64 x 64
        self.up2 = nc.VAEResnetBlock(64, 32, False) # 32 x 128 x 128

        self.res3 = nc.NVAEResBlocks(3, 32, 32)  # 32 x 128 x 128
        self.up3 = nc.VAEResnetBlock(32, 16, False) # 16 x 256 x 256

        self.res4 = nc.NVAEResBlocks(2, 16, 16)
        self.out = nn.Conv2d(16, 3, 3, 1, 1) # 3 x 256 x 256

    def forward(self, x):
        x = self.initial(x)
        # x = self.attnDec(x)
        x = self.res1(x)
        x = self.up1(x)
        x = self.res2(x)
        x = self.up2(x)
        x = self.res3(x)
        x = self.up3(x)
        x = self.res4(x)
        return F.tanh(self.out(x)) # force output to be between -1 & 1

class VAE(nn.Module):
    def __init__(self, latent_channels=8, dropout=0.0):
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