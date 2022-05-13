from turtle import forward
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3)):
        super(ConvBlock, self).__init__()
        self.layer = nn.Sequential(
                            nn.Conv2d(
                                in_channels,
                                out_channels,
                                kernel_size,
                                padding=(1, 1)
                                ),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.Conv2d(
                                out_channels,
                                out_channels,
                                kernel_size,
                                padding=(1, 1)
                                ),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU()
                            )

    def forward(self, x):
        return self.layer(x)

class UpBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 stride=(2, 1)) -> None:
        super().__init__()
        self.layer = nn.Sequential(
                            nn.ConvTranspose2d(
                                in_channels, 
                                out_channels, 
                                kernel_size,
                                stride=stride,
                                padding=(1, 0)
                            ),
                            nn.BatchNorm2d(out_channels), 
                            nn.ReLU()
                        )

    def forward(self, x):
        return self.layer(x)

class Model(nn.Module):
    def __init__(self,
                 num_upblocks=2,
                 in_channels=1,
                 dropout_rate=0,
                 channel_list=[64,
                               64*2,
                               64*4,
                               64*8,
                               64*16,
                               64*8]) -> None:
        super(Model, self).__init__()
        self.up_blocks = nn.ModuleList()

        for i in num_upblocks:
            self.up_blocks.append(UpBlock(
                                    in_channels, 
                                    channel_list[0]))

        self.conv_list = nn.ModuleList()
        self.conv_list.append(ConvBlock(
                                 channel_list[0],
                                 channel_list[1])
                             )

        for i in range(len(channel_list - 1)):
            self.conv_list.append(
                            nn.Sequential(
                                nn.AvgPool2d(
                                    kernel_size=(2, 2),
                                    stride=(2, 2),
                                    padding=(1, 1),
                                )),
                            nn.Dropout(dropout_rate), 
                            ConvBlock(
                                channel_list[i],
                                channel_list[i+1]
                            )
                        )
        