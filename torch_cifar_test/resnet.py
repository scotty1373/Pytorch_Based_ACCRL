# -*- coding: utf-8 -*-
import torch
import numpy as np
import os


class ResBlock(torch.nn.Module):
    def __init__(self, in_Channel, out_Channel, stride=1):
        super(ResBlock, self).__init__()
        # 第一层卷积步长使用给定步长，以便将图像长宽按stride倍数减半
        self.Conv1 = torch.nn.Conv2d(in_channels=in_Channel, out_channels=out_Channel, kernel_size=3, padding=1, stride=stride, bias=False)
        self.BatchNorm1 = torch.nn.BatchNorm2d(out_Channel)
        self.Conv2 = torch.nn.Conv2d(in_channels=out_Channel, out_channels=out_Channel, kernel_size=3, padding=1, stride=1, bias=False)
        self.BatchNorm2 = torch.nn.BatchNorm2d(out_Channel)
        # 为保证原输入通道数和输出通道数相同，使用kernal为1的卷积核改变输出channel5
        if in_Channel != out_Channel or stride != 1:
            self.Conv3 = torch.nn.Conv2d(in_channels=in_Channel, out_channels=out_Channel, kernel_size=1, stride=stride, bias=False)
        else:
            self.Conv3 = None

    def forward(self, input_):
        fx = self.Conv1(input_)
        fx = torch.nn.functional.relu(self.BatchNorm1(fx))
        fx = self.Conv2(fx)
        fx = self.BatchNorm2(fx)
        # 如果输入输出channel不匹配
        if self.Conv3:
            input_ = self.Conv3(input_)
        return torch.nn.functional.relu(fx + input_)    # 张量直接相加，不改变维度->128*14*14相加之后维度依然为128*14*14


class ResNet(torch.nn.Module):
    def __init__(self, channel_list, block_num):
        super(ResNet, self).__init__()
        self.image_channel = 3
        self.resnet_block = channel_list
        assert len(channel_list) == block_num

        self.net = self.net_initilize()

        self.layer1 = self.Block_Builder(64, 64, self.resnet_block[0], init=False)
        self.layer2 = self.Block_Builder(64, 128, self.resnet_block[1], init=True)      # 下采样，使用步长为2
        self.layer3 = self.Block_Builder(128, 256, self.resnet_block[2], init=True)     # 下采样，使用步长为2
        self.layer4 = self.Block_Builder(256, 512, self.resnet_block[3], init=True, stride=1)       # 步长为1，不改变大小

        self.Avgpool = torch.nn.AdaptiveAvgPool2d((4, 4))

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(512*4*4, 512),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(512, 10)
        )
        # x = torch.rand((3, 3, 32, 32))
        # out = self.net(x)
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        # print(out.shape)

    def forward(self, x):
        x = self.net(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.Avgpool(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc(x)
        return x

    @staticmethod
    def Block_Builder(inchannel, outchannel, num_block, init=False, stride=2):
        res_block = torch.nn.Sequential()
        for index in range(num_block):
            if init and index == 0:
                res_block.add_module("init_layer", ResBlock(inchannel,  outchannel, stride=stride))
            else:
                res_block.add_module(f"add_block{index}", ResBlock(outchannel, outchannel, stride=1))
        return res_block

    def net_initilize(self):
        net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.image_channel, out_channels=64, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU())
        # torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        return net


if __name__ == "__main__":
    net = ResNet([2, 2, 2, 2], 4)
    print(net)