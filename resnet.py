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
        # 为保证原输入通道数和输出通道数相同，使用kernal为1的卷积核改变输出channel
        if in_Channel != out_Channel and stride != 1:
            self.Conv3 = torch.nn.Conv2d(in_channels=in_Channel, out_channels=out_Channel, kernel_size=1, stride=stride, bias=False)

    def forward(self, input_):
        fx = self.Conv1(input_)
        fx = torch.nn.functional.relu(self.BatchNorm1(fx))
        fx = self.Conv2(fx)
        fx = self.BatchNorm2(fx)
        # 如果输入输出channel不匹配
        if self.Conv3:
            input_ = self.Conv3(input_)
        return torch.nn.functional.relu(fx + input_)


class ResNet(torch.nn.Module):
    def __init__(self, channel_list, block_num):
        super(ResNet, self).__init__()
        self.image_channel = channel_list
        self.resnet_block = block_num
        self.net = self.net_initilize()

    def forward(self, x):
        x = self.net(x)
        x

    def Block_Builder(self, inchannel, outchannel, init=False):
        res_block = torch.nn.Sequential()
        for index in range(self.resnet_block):
            if init and index == 1:
                res_block.add_module("init_layer", ResBlock(inchannel,  outchannel, stride=2))
            else:
                res_block.add_module(f"add_block{index}", ResBlock(inchannel, outchannel, stride=1))
        return res_block

    def net_initilize(self):
        net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.image_channel, out_channels=64, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        return net

