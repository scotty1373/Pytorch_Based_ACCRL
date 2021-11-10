# -*- coding: utf-8 -*-
import time

import torch
import torchvision
import numpy as np

batch_size = 128
device = torch.device('cpu')

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   # torchvision.transforms.Normalize(
                                   #     (0.1207,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

mnist_test = torchvision.datasets.MNIST('mnist_data', train=False, download=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            # torchvision.transforms.Normalize(
                                            #     (0.1207,), (0.3081,))
                                        ]))
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=mnist_test.data.shape[0], shuffle=False)

x, y = next(iter(train_loader))
print(x.shape, y.shape)


class Net_Builder(torch.nn.Module):
    def __init__(self):
        super(Net_Builder, self).__init__()

        # self.Conv1 = torch.nn.Conv2d(1, 32, kernel_size=(3, 3),
        #                              stride=(1, 1))
        #
        # self.Conv2 = torch.nn.Conv2d(32, 64, kernel_size=(3, 3),
        #                              stride=(1, 1))
        # self.Conv3 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3),
        #                              stride=(1, 1))
        self.fc1 = torch.nn.Linear(28*28, 256)
        self.fc2 = torch.nn.Linear(256, 64)
        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def training():
    global net
    train_loss_ = np.zeros((3, 60000//batch_size))
    start_time = time.time()
    for epoch in range(3):
        # the parameter which need to update, put the parameter in optimizer's parameter option
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

        for batch_idx, (x_train, y_train) in enumerate(train_loader):
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            x_train = x_train.view(-1, 28*28)
            out = net(x_train)
            y_onehot = torch.nn.functional.one_hot(y_train, 10)
            # loss
            loss = torch.nn.functional.mse_loss(out, y_onehot.float())

            optimizer.zero_grad()
            loss.backward()
            # w' = w - lr * grad_parameter
            optimizer.step()
            train_loss_[epoch, batch_size] = loss
            if batch_idx % 10 == 0:
                print(f'epoch: {epoch}, batch: {batch_idx}, loss:{loss}')

        single_label_acc = np.zeros((10, 1))
        for test_idx, (x_test, y_test) in enumerate(test_loader):
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            for labels in range(10):
                single_labels_index = np.squeeze(np.argwhere(y_test.cpu().numpy() == labels))
                single_labels_x_test = x_test[single_labels_index, :, :, :].view(-1, 28*28)
                result = net(single_labels_x_test)
                result = torch.argmax(result, dim=1)
                correct = torch.eq(result, y_test[single_labels_index])
                single_label_acc[labels, :] = torch.mean(correct.float()).cpu()
                print(f'Test acc of num {labels}: {single_label_acc[labels, :]}')
        print(f'loss mean: {np.mean(train_loss_[epoch, :])}')
    print(f'time cost: {time.time() - start_time}')


if __name__ == '__main__':
    net = Net_Builder().to(device)
    training()
