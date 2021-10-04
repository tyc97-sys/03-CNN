# Import需要的套件
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
from readfile import *
from module import *

random.seed(1024)



# 分別將 training set、validation set、testing set 用 readfile 函式讀進來

workspace_dir = './food-11'

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


# Model


model = Classifier().cuda()
loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # optimizer 使用 Adam
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # optimizer with SGDM

num_epoch = 10

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()  # 確保 model 是在 train model (開啟 Dropout 等...)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()  # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = model(data[0].cuda())  # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].cuda())  # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward()  # 利用 back propagation 算出每個參數的 gradient
        optimizer.step()  # 以 optimizer 用 gradient 更新參數值

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        # 將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
              (epoch + 1, num_epoch, time.time() - epoch_start_time, \
               train_acc / train_set.__len__(), train_loss / train_set.__len__(), val_acc / val_set.__len__(),
               val_loss / val_set.__len__()))
# 得到好的參數後，我們使用 training set 和 validation set 共同訓練（資料量變多，模型效果較好）
batch_size = 64
train_val_x = np.concatenate((train_x, val_x), axis=0)
train_val_y = np.concatenate((train_y, val_y), axis=0)
train_val_set = ImgDataset(train_val_x, train_val_y, None)
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)


# model_best = Classifier().cuda()
# loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
# # optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001) # optimizer 使用 Adam
# optimizer = torch.optim.SGD(model_best.parameters(), lr=0.001, momentum=0.9)   # optimizer with SGDM
# num_epoch = 10
# for epoch in range(num_epoch):
#     epoch_start_time = time.time()
#     train_acc = 0.0
#     train_loss = 0.0
#
#     model_best.train()
#     for i, data in enumerate(train_val_loader):
#         optimizer.zero_grad()
#         train_pred = model_best(data[0].cuda())
#         batch_loss = loss(train_pred, data[1].cuda())
#         batch_loss.backward()
#         optimizer.step()
#
#         train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
#         train_loss += batch_loss.item()
#
#         # 將結果 print 出來
#     print('[%03d/%03d] %2.2f sec(s) Train Acc:+ %3.6f Loss: %3.6f' % \
#           (epoch + 1, num_epoch, time.time() - epoch_start_time, \
#            train_acc / train_val_set.__len__(), train_loss / train_val_set.__len__()))

# saving model
PATH_model = './model_best_ADAM_10'
PATH_model += '.pkl'
torch.save(model, PATH_model)

# getting confusion matrix on valid set
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

