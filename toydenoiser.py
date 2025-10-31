import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

class GrayscaleChurches(Dataset):

    def __init__(self, file_dir="churches.npy", training=True):
        super().__init__()
        if training:
            self.data = np.load(file_dir)[:70000] # Limit training data
        else:
            self.data = np.load(file_dir)[70000:]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        # Retrieve an item and convert it to a form ready for training
        # Format: (channels, rows, cols)
        im = self.data[item]

        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        im = im / 255 * 2 - 1 # Scale into the interval [-1, 1]
        im = torch.tensor(im, dtype=torch.float32).view(1, 64, 64)

        noise = 0.05 * torch.randn_like(im)
        return im + noise, noise

    def get_clean(self, item):
        return self.data[item]

class NoiseFinder(nn.Module):

    def __init__(self):
        super(NoiseFinder, self).__init__()

        # Downward pass of the UNet
        self.down1 = ConvolutionBlock([1, 32, 32], 64, 64)
        self.down2 = ConvolutionBlock([32, 64, 64], 32, 32)
        self.down3 = ConvolutionBlock([64, 128, 128], 16, 16)
        self.down4 = ConvolutionBlock([128, 256, 256], 8, 8)

        # Upward pass of the UNet
        self.upconv1 = nn.ConvTranspose2d(256, 128, 2, 2, 0)
        self.up1 = ConvolutionBlock([256, 128, 128], 16, 16)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, 2, 0)
        self.up2 = ConvolutionBlock([128, 64, 64], 32, 32)
        self.upconv3 = nn.ConvTranspose2d(64, 32, 2, 2, 0)
        self.up3 = ConvolutionBlock([64, 32, 32], 64, 64)

        # Final 1x1 convolution to map to a single grid
        self.one = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # Perform downward pass
        res1 = self.down1(x) # 32 x 64 x 64
        x = F.max_pool2d(res1, 2, 2)
        res2 = self.down2(x) # 64 x 32 x 32
        x = F.max_pool2d(res2, 2, 2)
        res3 = self.down3(x) # 128 x 16 x 16
        x = F.max_pool2d(res3, 2, 2)
        res4 = self.down4(x) # 256 x 8 x 8

        # Do upward pass, adding in residual skip connections along the way
        x = self.upconv1(res4) # 128 x 16 x 16
        x = torch.cat([res3, x], dim=1) # 256 x 16 x 16
        x = self.up1(x) # 128 x 16 x 16

        x = self.upconv2(x) # 64 x 32 x 32
        x = torch.cat([res2, x], dim=1) # 128 x 32 x 32
        x = self.up2(x) # 64 x 32 x 32

        x = self.upconv3(x) # 32 x 64 x 64
        x = torch.cat([res1, x], dim=1) # 64 x 64 x 64
        x = self.up3(x) # 32 x 64 x 64

        return self.one(x)


class ConvolutionBlock(nn.Module):

    def __init__(self, channel_sequence, rows, cols):
        super(ConvolutionBlock, self).__init__()

        self.conv1 = nn.Conv2d(channel_sequence[0], channel_sequence[1], 3, 1, 1)
        self.conv2 = nn.Conv2d(channel_sequence[1], channel_sequence[2], 3, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


net = NoiseFinder()

# Training code...
dat = GrayscaleChurches()
epochs = 5
batch_size = 32

optimizer = torch.optim.Adam(net.parameters(), 0.001)
loss_function = nn.MSELoss()

for epoch in range(epochs):
    dat_loader = DataLoader(dat, batch_size, shuffle=True)
    progress_bar = tqdm(dat_loader, desc=f"Epoch {epoch+1}:")

    for _, batch in enumerate(progress_bar):
        x, y = batch

        optimizer.zero_grad()

        logits = net(x)

        loss = loss_function(logits, y)
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix({"Loss":loss.item()})

    torch.save(net.state_dict(), f"denoiser{epoch+1}.pt")

# Testing code
# dat = GrayscaleChurches(training=False)
# net.load_state_dict(torch.load("denoiser1.pt"))
#
# with torch.no_grad():
#     for noisy, noise in dat:
#
#         fig, ax = plt.subplots(1, 4)
#         ax[0].imshow(noisy[0])
#         ax[0].axis('off')
#         ax[0].set_title("Image w/ Noise")
#         ax[1].imshow(noise[0])
#         ax[1].axis('off')
#         ax[1].set_title("Noise")
#         pred_noise = net(noisy.view(1,1,64,64))
#         ax[2].imshow(pred_noise[0,0])
#         ax[2].axis('off')
#         ax[2].set_title("Predicted Noise")
#         ax[3].imshow(pred_noise[0,0]-noise[0])
#         ax[3].axis('off')
#         ax[3].set_title("Difference")
#
#         plt.show()