from turtle import forward
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3)):
        super(ConvBlock, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_channels,
                                             out_channels,
                                             kernel_size,
                                             padding=(1, 1)),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU(),
                                   nn.Conv2d(out_channel,
                                             out_channel,
                                             kernel_size,
                                             padding=(1, 1)),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU())

    def forward(self, x):
        return self.layer(x)

class UpBlock(nn.Module):
    def __init__(self,
                 in_channels,) -> None:
        super().__init__()

class Model(nn.Module):
    def __init__(self,
                 channel_list=[64,
                               64*2,
                               64*4,
                               64*8,
                               64*16,
                               64*8]) -> None:
        super(Model, self).__init__()
        self.conv