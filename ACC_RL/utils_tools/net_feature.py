# -*- coding: utf-8 -*-
import torch
import os
import numpy as np

class yolo_feature_processor(torch.nn.Module):
    def __init__(self, innershape, outshape):
        super(yolo_feature_processor, self).__init__()
        self.Dense1 = torch.nn.Linear(innershape, 32)
        self.activation1 = torch.nn.ReLU(inplace=True)
        self.Dense2 = torch.nn.Linear(32, 64)
        self.activation2 = torch.nn.ReLU(inplace=True)
        self.inputDense1 = torch.nn.Linear(4, 32)
        self.activation_iD = torch.nn.ReLU(inplace=True)
        self.catDense1 = torch.nn.Linear(96, 32)
        self.catactivation = torch.nn.ReLU(inplace=True)
        self.catDense2 = torch.nn.Linear(32, outshape)

    def forward(self, x_, y_):
        out = self.Dense1(x_)
        out = self.activation1(out)
        out = self.Dense2(out)
        out = self.activation2(out)
        alterout = self.inputDense1(y_)
        alterout = self.activation_iD(alterout)
        concat_vector = torch.cat([out, alterout], dim=1)
        concat_vector = self.catDense1(concat_vector)
        concat_vector = self.catactivation(concat_vector)
        concat_vector = self.catDense2(concat_vector)
        return concat_vector


if __name__ == "__main__":
    x = torch.randn((1, 4))
    y = torch.randn((1, 4))

    print(x, y)

    model = yolo_feature_processor(4, 21)

    result = model(x, y)
    print(result)
