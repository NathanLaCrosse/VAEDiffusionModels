import torch
import os
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class ChurchData(nn.Module):

    def __init__(self, file_dir="churches.npy"):
        super().__init__()
        self.data = np.load(file_dir)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        # Retrieve an item and convert it to a form ready for training
        # Format: (channels, rows, cols)
        return torch.tensor(self.data[item]).permute(2, 0, 1)

    def get_cleaned(self, item):
        # Get an unedited form of an image
        return self.data[item]


dat = ChurchData()

a = dat[0]

print("hi")