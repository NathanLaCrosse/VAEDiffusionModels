import torch
import os
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import cv2
import matplotlib.pyplot as plt

def generate_train_test_split(data_dir="MushroomData/", train_prop=0.8, prefix=""):
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
    with open(f'DataJsons/class2idx.json', 'r') as file:
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
    with open(f"DataJsons/{prefix}traindirs.json", 'w') as file:
        json.dump(train_data, file)
    with open(f"DataJsons/{prefix}testdirs.json", 'w') as file:
        json.dump(test_data, file)

    return train_data, test_data


class MushroomData(Dataset):
    def __init__(self, json_file, mse_mode=False, prefix="", halve=False):
        super(MushroomData, self).__init__()

        self.mse_mode = mse_mode
        self.halve = halve

        with open(json_file, 'r') as file:
            self.data = json.load(file)

        self.prefix = prefix

        # self.data = self.data[:1000]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        path, label = self.data[item]

        im = cv2.imread(self.prefix + path)[:,:,::-1]

        if self.halve:
            im = cv2.resize(im, (im.shape[0]//2, im.shape[1]//2), interpolation=cv2.INTER_AREA)

        im = im / 255  # Scale to be in [0, 1]
        if self.mse_mode:
            im = im * 2 - 1 # Scale to be in [-1, 1]
        im = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1).contiguous()

        return im, label

    def get_clean(self, item):
        path, label = self.data[item]

        im = cv2.imread(self.prefix + path)[:,:,::-1]

        if self.halve:
            im = cv2.resize(im, (im.shape[0]//2, im.shape[1]//2), interpolation=cv2.INTER_AREA)

        return im, label

if __name__ == "__main__":
    # smol, _ = generate_train_test_split(data_dir="MushroomData/", train_prop=1, prefix="sixtyfour")
    # big, _ = generate_train_test_split(data_dir="CleanedData/", train_prop=1, prefix="twofiftysix")
    #
    # collected = []
    # print("Collecting...")
    #
    # big_dict = {}
    # for i in range(len(big)):
    #     big_dict[big[i][0]] = i
    #
    # for key, val in smol:
    #     file_name = "CleanedData/" + key[13:]
    #
    #     found = False
    #     try:
    #         found = big_dict[file_name] is not None
    #     except:
    #         pass
    #
    #     if found:
    #         collected.append([key[13:], val])
    #
    # # We want to convert indices to something continuous.
    # map = {}
    #
    # increment = 0
    # for i in range(len(collected)):
    #     val = collected[i][1]
    #
    #     if val not in map.keys():
    #         map[val] = increment
    #         increment += 1
    #
    # for i in range(len(collected)):
    #     collected[i][1] = map[collected[i][1]]
    #
    # with open(f"DataJsons/cleaningshift.json", 'w') as file:
    #     json.dump(map, file)
    #
    # with open(f"DataJsons/combineddirs.json", 'w') as file:
    #     json.dump(collected, file)
    #
    # print(len(collected))

    # dat = MushroomData("DataJsons/combineddirs.json", True, "MushroomData/")
    dat = MushroomData("DataJsons/combineddirs.json", True, "CleanedData/", halve=True)

    for i in range(len(dat)):
        plt.imshow(dat.get_clean(np.random.randint(0, len(dat)))[0])
        plt.show()

