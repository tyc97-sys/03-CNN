import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
import random
random.seed(1024)
def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img, (128, 128))
        if label:
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x


def make_set(path):
    print("Reading data")
    train_x, train_y = readfile(os.path.join(path, "training"), True)
    print("Size of training data = {}".format(len(train_x)))
    val_x, val_y = readfile(os.path.join(path, "validation"), True)
    print("Size of validation data = {}".format(len(val_x)))
    test_x = readfile(os.path.join(path, "testing"), False)
    print("Size of Testing data = {}".format(len(test_x)))

    return train_x, train_y, val_x, val_y, test_x