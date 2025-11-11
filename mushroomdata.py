import torch
import os
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import cv2
import matplotlib.pyplot as plt

def generate_train_test_split(data_dir="MushroomData/", train_prop=0.8):
    """
    Generate a train test split of directories from data_dir.
    These directory lists will be saved to two separate json files as lists.
    These lists consist of (path, label) pairs.
    :param data_dir: The folder directory of the data
    :param train_prop: The proportion of the data to be put in the train set.
    """
    
    train_data = []
    test_data = []
    
    # Load class 2 idx for saving classifications
    with open('DataJsons/class2idx.json', 'r') as file:
        class2idx = json.load(file)
    
    # Traverse the dataset
    for folder_dir in os.listdir(data_dir):
        # Collect all samples from one classification of mushroom
        classifications = [None] * len(os.listdir(data_dir + folder_dir))

        i = 0
        for file_dir in os.listdir(data_dir + folder_dir):
            path = data_dir + folder_dir + "/" + file_dir
            classifications[i] = path
            i += 1

        classifications = np.array(classifications)
        np.random.shuffle(classifications)

        partition = int(train_prop * len(classifications))

        for val in classifications[:partition]:
            train_data.append((val, class2idx[folder_dir]))
        for val in classifications[partition:]:
            test_data.append((val, class2idx[folder_dir]))

    # Save the train and test
    with open("DataJsons/traindirs.json", 'w') as file:
        json.dump(train_data, file)
    with open("DataJsons/testdirs.json", 'w') as file:
        json.dump(test_data, file)


class MushroomData(Dataset):
    def __init__(self, json_file, mse_mode=False):
        super(MushroomData, self).__init__()

        self.mse_mode = mse_mode

        with open(json_file, 'r') as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        path, label = self.data[item]

        im = cv2.imread(path)[:,:,::-1]
        im = im / 255  # Scale to be in [0, 1]
        if self.mse_mode:
            im = im * 2 - 1 # Scale to be in [-1, 1]
        im = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1).contiguous()

        return im, label

    def get_clean(self, item):
        path, label = self.data[item]
        return cv2.imread(path)[:,:,::-1], label

if __name__ == "__main__":
    # generate_train_test_split()

    dat = MushroomData("DataJsons/traindirs.json")

    for i in range(len(dat)):
        plt.imshow(dat.get_clean(np.random.randint(0, len(dat)))[0])
        plt.show()

