import torch
import os
from torch import nn
import numpy as np

class ChurchData(nn.Module):

    def __init__(self, file_dir="churches.npy"):
        super().__init__()
        self.data = np.load(file_dir)[:700000] # Limit training data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        # Retrieve an item and convert it to a form ready for training
        # Format: (channels, rows, cols)
        im = self.data[item]
        im = im / 255 * 2 - 1 # Scale into the interval [-1, 1]
        return torch.tensor(im).permute(2, 0, 1)

    def get_clean(self, item):
        return self.data[item]


dat = ChurchData()

a = dat[0]
b = dat.get_clean(0)

print("hi")