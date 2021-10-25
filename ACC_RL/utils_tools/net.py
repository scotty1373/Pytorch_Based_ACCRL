# -*- coding: utf-8 -*-
import os
import torch
from torch import nn
import itertools


class Common(nn.Module):
    def __init__(self):
        super(Common, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16,
                               kernel_size=(8, 8), stride=(4, 4))
        self.actv1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=(4, 4), stride=(4, 4),
                               padding=(1, 1))
        self.actv2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.actv3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.actv4 = nn.ReLU(inplace=True)

        self.Dense1 = nn.Linear(3200, 512)
        self.actvDense1 = nn.ReLU(inplace=True)
        self.Dense2 = nn.Linear(512, 128)
        self.actvDense2 = nn.ReLU(inplace=True)
        self.Dense3 = nn.Linear(128, 64)

        self.inputDense1 = nn.Linear(4, 16)
        self.actvinput = nn.ReLU(inplace=True)
        self.inputDense2 = nn.Linear(16, 64)

    def forward(self, x1, x2):
        feature = self.conv1(x1)
        feature = self.actv1(feature)
        feature = self.conv2(feature)
        feature = self.actv2(feature)
        feature = self.conv3(feature)
        feature = self.actv3(feature)
        feature = self.conv4(feature)
        feature = self.actv4(feature)
        feature = torch.flatten(feature, start_dim=1, end_dim=-1)
        feature = self.Dense1(feature)
        feature = self.actvDense1(feature)
        feature = self.Dense2(feature)
        feature = self.actvDense2(feature)
        feature = self.Dense3(feature)

        out = self.inputDense1(x2)
        out = self.actvinput(out)
        out = self.inputDense2(out)

        output = torch.cat([feature, out], dim=1)
        return output


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        # self.common = Common()
        self.Dense1 = nn.Linear(128, 32)
        self.actv1 = nn.ReLU(inplace=True)
        self.Dense2 = nn.Linear(32, 1)
        torch.nn.init.uniform_(self.Dense2.weight, a=-3e-3, b=3e-3)
        self.actv2 = nn.Tanh()

    def forward(self, common):
        action = self.Dense1(common)
        action = self.actv1(action)
        action = self.Dense2(action)
        action = self.actv2(action)

        return action


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        # self.common = Common()
        self.Dense1 = nn.Linear(128, 32)
        self.actv1 = nn.ReLU(inplace=True)
        self.inputDense = nn.Linear(1, 16)
        self.Dense2 = nn.Linear(48, 1)
        torch.nn.init.uniform_(self.Dense2.weight, a=-3e-3, b=3e-3)

    def forward(self, common, action):
        critic = self.Dense1(common)
        critic = self.actv1(critic)
        input_action = self.inputDense(action)
        critic_out = torch.cat([critic, input_action], dim=1)
        critic_out = self.Dense2(critic_out)
        return critic_out


if __name__ == '__main__':
    common_net = Common()
    actor_net = Actor()
    critic_net = Critic()
    loss = torch.nn.MSELoss()
    opt_common = torch.optim.SGD(common_net.parameters(), 0.001)
    # opt_actor = torch.optim.SGD(itertools.chain(common_net.parameters(), actor_net.parameters()), 0.001)
    opt_actor = torch.optim.SGD(actor_net.parameters(), 0.001)
    opt_critic = torch.optim.SGD(itertools.chain(common_net.parameters(), actor_net.parameters()), 0.001)
    print(common_net)
    print(actor_net)
    print(critic_net)

    x = torch.randn((10, 4, 80, 80))
    y = torch.randn((10, 4))

    out1 = actor_net(common_net(x, y))
    out2 = critic_net(common_net(x, y), out1)

    tgt = torch.rand(10, 1)

    loss_scale = loss(out1, tgt)
    opt_actor.zero_grad()
    loss_scale.backward()
    opt_actor.step()



