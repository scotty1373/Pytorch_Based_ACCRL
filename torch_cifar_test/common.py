# -*- coding: utf-8 -*-
import torch
from cifar10_dataset_collect import dataset_collect
from Lenet import Lenet as ln
from resnet import ResNet
import numpy as np

batch_size = 64
epoches = 100
num_cate = 10

cifar_train, cifar_test = dataset_collect(batchsz=batch_size)


def train():
    loss_history = []
    acc_history = []
    for epoch in range(epoches):
        for bsindex, (train_x, train_y) in enumerate(cifar_train):
            train_x, train_y = train_x.to(device), train_y.to(device)
            logits = model(train_x)
            loss = criterion(logits, train_y)
            loss_history.append(loss)

            # backprop
            optim.zero_grad()   # 初始化梯度,不清零的话每次optimizer的梯度是进行累加计算
            loss.backward()
            optim.step()
            if bsindex // 20 == 0:
                print(f'epoch: {epoch}, loss: {loss}')
        for bsindex, (test_x, test_y) in enumerate(cifar_test):
            test_x, test_y = test_x.to(device), test_y.to(device)
            logits = model(test_x)
            pred = torch.nn.functional.softmax(logits, dim=1)
            pred = torch.argmax(pred, dim=1).squeeze()
            currect_cate = torch.mean(torch.eq(pred, test_y).float())
            acc_history.append(currect_cate)
        print(f"Epoch: {epoch}, Accuracy: {np.array(acc_history).mean()}")


if __name__ == "__main__":
    device = torch.device('cpu')
    model = ResNet([2, 2, 2, 2], 4).to(device)
    print(model)    # 打印类实例
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    train()




