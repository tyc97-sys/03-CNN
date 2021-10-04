# Import需要的套件
import time
from module import *
from train import ImgDataset

model_best = torch.load('model1.pth')
print("loading")

prediction = []

train_x, train_y, val_x, val_y, test_x = make_data_set(data_path)

test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model_best.eval()

with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model_best(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            prediction.append(y)

# 將結果寫入 csv 檔
with open("predict.csv", 'w') as f:
    f.write('Id,Category\n')
    for i, y in enumerate(prediction):
        f.write('{},{}\n'.format(i, y))

model.eval()
prediction = []
ground_truth = []
with torch.no_grad():
    for i, (data, ans) in enumerate(val_loader):
        test_pred = model(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y, gt in zip(test_label,ans):
            prediction.append(y)
            ground_truth.append(gt.cpu().item())

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=ground_truth, y_pred=prediction, normalize='true')
confmat = np.round(confmat, decimals = 3)
fig, ax = plt.subplots(figsize=(25, 25))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center', fontsize = 40 )
plt.xticks(size = 25)
plt.yticks(size = 25)
plt.xlabel('predicted label', fontsize = 40)
plt.ylabel('true label', fontsize = 40)