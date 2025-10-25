import numpy as np
import cv2
import matplotlib.pyplot as plt

dat = np.load("churches.npy")

# View each image - displays a new one once window is closed.
for i in range(len(dat)):
    plt.imshow(dat[i])
    plt.show()