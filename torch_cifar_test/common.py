# -*- coding: utf-8 -*-
import torch
from cifar10_dataset_collect import dataset_collect
from Lenet import Lenet as ln
from resnet import ResNet

batch_size = 64
epoches = 100
num_cate = 10

cifar_train, cifar_test = dataset_collect(batchsz=batch_size)


def train():
    loss_history = []
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


if __name__ == "__main__":
    model = ResNet([2, 2, 2, 2], 4)
    device = torch.device('cpu')
    print(model)    # 打印类实例
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    train()




