import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


def dataset_collect(batchsz):
    cifar_train = datasets.CIFAR10('cifar', train=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz[0], shuffle=True, num_workers=12)

    cifar_test = datasets.CIFAR10('cifar_test', train=False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz[1], shuffle=True, num_workers=12)

    # test
    # x, x_label = iter(cifar_train)
    # print(x.shape, x_label.shape)
    # y, y_label = iter(cifar_test)
    # print(y.shape, y_label.shape)
    return cifar_train, cifar_test


if __name__ == "__main__":
    dataset_collect(64)