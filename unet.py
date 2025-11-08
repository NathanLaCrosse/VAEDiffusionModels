import torch
from torch import nn
import torch.nn.functional as F
import NetworkComponents as nc
import numpy as np

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
            nn.Linear(time_embed_dim * 4, time_embed_dim),
            nn.SiLU()
        )
        self.label_mlp = nn.Sequential(
            nn.Embedding(num_classes, label_embed_dim),
            nn.Linear(label_embed_dim, label_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(label_embed_dim*4, label_embed_dim),
            nn.SiLU()
        )

        # Downward pass of the UNet
        self.initial = nn.Conv2d(8, 16, 1) # 8 x 8 x 8 -> 16 x 8 x 8
        self.downres1 = nc.ResidualBlockWithEmbeddings(16, 8, 8, time_embed_dim=time_embed_dim,
                                                   label_embed_dim=label_embed_dim, dropout_p=dropout_p) # 16 x 8 x 8
        self.down1 = nn.Conv2d(16, 32, 3, 2, 1) # 16 x 8 x 8 -> 32 x 4 x 4
        self.downres2 = nc.ResidualBlockWithEmbeddings(32, 16, 4, time_embed_dim=time_embed_dim,
                                                   label_embed_dim=label_embed_dim, dropout_p=dropout_p) # 32 x 4 x 4
        self.down2 = nn.Conv2d(32, 64, 3, 2, 1) # 32 x 4 x 4 -> 64 x 2 x 2
        self.downres3 = nc.ResidualBlockWithEmbeddings(64, 32, 2, time_embed_dim=time_embed_dim,
                                                   label_embed_dim=label_embed_dim, dropout_p=dropout_p) # 64 x 2 x 2

        # Upward pass of the UNet
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, 2) # 64 x 2 x 2 -> 32 x 4 x 4
        # Concatenation happens here -> # 64 x 4 x 4 (in forward method)
        self.reduce_channels1 = nn.Conv2d(64, 32, 1) # 64 x 4 x 4 -> 32 x 4 x 4
        self.upres1 = nc.ResidualBlockWithEmbeddings(32, 16, 4, time_embed_dim=time_embed_dim,
                                                     label_embed_dim=label_embed_dim, dropout_p=dropout_p)

        self.upconv2 = nn.ConvTranspose2d(32, 16, 2, 2) # 32 x 4 x 4 -> 16 x 8 x 8
        # Concatenation happens here -> 32 x 8 x 8
        self.reduce_channels2 = nn.Conv2d(32, 16, 1) # 32 x 8 x 8 -> 16 x 8 x 8
        self.upres2 = nc.ResidualBlockWithEmbeddings(16, 8, 8, time_embed_dim=time_embed_dim,
                                                     label_embed_dim=label_embed_dim, dropout_p=dropout_p)

        self.to_out = nn.Conv2d(16, 8, 1)

    def forward(self, x, time_embed, l):
        global_t = self.time_mlp(time_embed)
        global_l = self.label_mlp(l)

        step1 = self.initial(x) # 8 x 8 x 8 -> 16 x 8 x 8
        step1 = self.downres1(step1, global_t, global_l) # 16 x 8 x 8 -> 16 x 8 x 8

        step2 = self.down1(step1) # 16 x 8 x 8 -> 32 x 4 x 4
        step2 = self.downres2(step2, global_t, global_l) # 32 x 4 x 4 -> 32 x 4 x 4

        step3 = self.down2(step2) # 32 x 4 x 4 -> 64 x 2 x 2
        step3 = self.downres3(step3, global_t, global_l) # 64 x 2 x 2 -> 64 x 2 x 2 (Bottom step)

        up = self.upconv1(step3) # 64 x 2 x 2 -> 32 x 4 x 4
        up = torch.cat([step2, up], dim=1) # 32 x 4 x 4 -> 64 x 4 x 4
        up = self.reduce_channels1(up) # 64 x 4 x 4 -> 32 x 4 x 4
        up = self.upres1(up, global_t, global_l) # 32 x 4 x 4 -> 32 x 4 x 4

        up = self.upconv2(up) # 32 x 4 x 4 -> 16 x 8 x 8
        up = torch.cat([step1, up], dim=1) # 16 x 8 x 8 -> 32 x 8 x 8
        up = self.reduce_channels2(up) # 32 x 8 x 8 -> 16 x 8 x 8
        up = self.upres2(up, global_t, global_l) # 16 x 8 x 8 -> 16 x 8 x 8

        return self.to_out(up)

if __name__ == '__main__':
    test_x = torch.randn(2,8,8,8)
    test_t = torch.randn(2,10)
    test_l = torch.randint(7, (2,))

    unet = UNET(10, 13, 7)

    pred = unet(test_x, test_t, test_l)

    print('hi')
