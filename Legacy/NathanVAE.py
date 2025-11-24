import torch
from torch import nn
import torch.nn.functional as F
import NetworkComponents as nc

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

    def forward_encode_only_reparam(self, x):
        x, _ = self.encoder(x)
        return x

    def forward_encode_only_mean(self, x):
        x, _ = self.encoder.forward_to_mean_std(x)
        return x