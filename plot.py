# getting confusion matrix on valid set
# Import需要的套件
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import time
import random
from readfile import *
from module import *
from ImgDataset import *

model = torch.load('model_best_ADAM_10.pkl')

workspace_dir = r'E:\Dataset\CNN_dataset\food-11'

train_x, train_y, val_x, val_y, test_x = make_set(workspace_dir)

# training 時做 data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),  # 隨機將圖片水平翻轉
    transforms.RandomRotation(15),  # 隨機旋轉圖片
    transforms.ToTensor(),  # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
])
# testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

batch_size = 64
train_set = ImgDataset(train_x, train_y, train_transform)
# train_set = ImgDataset(train_x, train_y, None)
val_set = ImgDataset(val_x, val_y, test_transform)
# val_set = ImgDataset(val_x,val_y, None)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


model.eval()
prediction = []
ground_truth = []
with torch.no_grad():
    for i, (data, ans) in enumerate(val_loader):
        test_pred = model(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y, gt in zip(test_label, ans):
            prediction.append(y)
            ground_truth.append(gt.cpu().item())

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

confmat = confusion_matrix(y_true=ground_truth, y_pred=prediction, normalize='true')
confmat = np.round(confmat, decimals=3)
fig, ax = plt.subplots(figsize=[13, 13])
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.xticks(size=10)
plt.yticks(size=10)
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()