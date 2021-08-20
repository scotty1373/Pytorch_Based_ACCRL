# -*- coding: utf-8 -*-
from torch import nn

class Data_dim_reduce:
    def __init__(self):
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(8, 8),
                               stride=(4, 4))
        self.activation1 = nn.ReLU()
        # 在keras中，通过卷积计算的输出图像如果不为整数，则只取整数部分，相当于留有未卷积的边
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4),
                               stride=(4, 4), padding=)