# -*- coding: utf-8 -*-
import time

import torch
from cifar10_dataset_collect import dataset_collect
from Lenet import Lenet as ln
from resnet import ResNet
import numpy as np
import matplotlib.pyplot as plt

batch_size = [64, 1000]
epoches = 100
num_cate = 10

cifar_train, cifar_test = dataset_collect(batchsz=batch_size)


def train():
    loss_history = []
    acc_history = []
    for epoch in range(epoches):
        loss_cache = []
        acc_cache = []
        for bsindex, (train_x, train_y) in enumerate(cifar_train):
            train_x, train_y = train_x.to(device), train_y.to(device)
            logits = model(train_x)
            loss = criterion(logits, train_y)
            loss_cache.append(loss.cpu().item())

            # backprop
            optim.zero_grad()   # 初始化梯度,不清零的话每次optimizer的梯度是进行累加计算
            loss.backward()
            optim.step()
            if bsindex // 20 == 0:
                print(f'epoch: {epoch}, loss: {loss}')
        loss_history.append(np.array(loss_cache).mean())
        start_time = time.time()
        for bsindex, (test_x, test_y) in enumerate(cifar_test):
            test_x, test_y = test_x.to(device), test_y.to(device)
            logits = model(test_x)
            pred = torch.nn.functional.softmax(logits, dim=1)
            pred = torch.argmax(pred, dim=1).squeeze()
            currect_cate = torch.mean(torch.eq(pred, test_y).float())
            acc_cache.append(currect_cate.cpu().item())
        acc_history.append(np.array(acc_cache).mean())
        print(f"Epoch: {epoch}, Accuracy: {np.array(acc_history).mean()}, time_cost: {(time.time()-start_time)*1000} ms")

        # 更新学习率
        scheduler.step()

    pl1, = plt.plot(np.arange(len(loss_history)), np.array(loss_history), color='b', linestyle=':')
    pl2, = plt.plot(np.arange(len(acc_history)), np.array(acc_history), color='g')
    plt.legend(handles=[pl1, pl2], labels=['loss', 'acc'], loc='best')
    plt.show()


if __name__ == "__main__":
    device = torch.device('cpu')
    model = ResNet([2, 2, 2, 2], 4).to(device)
    print(model)    # 打印类实例
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    # 根据步长变化学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=0.1, last_epoch=-1)
    criterion = torch.nn.CrossEntropyLoss()
    train()




