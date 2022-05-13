from turtle import forward
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 padding=(1, 1)):
        super(ConvBlock, self).__init__()
        self.layer = nn.Sequential(
                            nn.Conv2d(
                                in_channels,
                                out_channels,
                                kernel_size,
                                padding=padding
                                ),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.Conv2d(
                                out_channels,
                                out_channels,
                                kernel_size,
                                padding=padding
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
                 stride=(2, 1),
                 padding=(1, 1),
                 out_padding=(1, 0)) -> None:
        super().__init__()
        self.layer = nn.Sequential(
                            nn.ConvTranspose2d(
                                in_channels, 
                                out_channels, 
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                output_padding=out_padding
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
                 encoder_channel_list=[64, 64*2, 64*4,
                               64*8, 64*16],
                 decoder_channel_list=[64*16, 64*8, 64*4,
                                       64*2, 64]) -> None:
        super(Model, self).__init__()
        
        self.up_blocks = nn.ModuleList()
        self.encoder_list = nn.ModuleList()
        self.decoder_list = nn.ModuleList()

        for i in range(num_upblocks):
            self.up_blocks.append(UpBlock(
                                    (in_channels if i == 0 
                                     else encoder_channel_list[0]), 
                                     encoder_channel_list[0]))

        self.encoder_list.append(ConvBlock(
                                 encoder_channel_list[0],
                                 encoder_channel_list[0])
                             )

        for i in range(len(encoder_channel_list) - 1):
            self.encoder_list.append(
                            nn.Sequential(
                                nn.AvgPool2d(
                                    kernel_size=(2, 2),
                                    stride=(2, 2),
                                    padding=(0, 0)),
                                nn.Dropout(dropout_rate), 
                                ConvBlock(
                                    encoder_channel_list[i],
                                    encoder_channel_list[i+1])
                                )
                            )
        
        for i in range(len(decoder_channel_list)-1):
            if i == 0:
                self.decoder_list.append(
                            nn.Sequential(
                                nn.Dropout(dropout_rate), 
                                UpBlock(
                                    decoder_channel_list[i],
                                    decoder_channel_list[i+1], 
                                    stride=(2, 2),
                                    padding=(1, 1),
                                    out_padding=(1, 1))
                            ))
                continue

            self.decoder_list.append(
                            nn.Sequential(
                                ConvBlock(
                                    decoder_channel_list[i]*2,
                                    decoder_channel_list[i+1]*2),
                                nn.Dropout(dropout_rate), 
                                UpBlock(
                                    decoder_channel_list[i],
                                    decoder_channel_list[i+1], 
                                    stride=(2, 2),
                                    padding=(1, 1),
                                    out_padding=(1, 1))
                            ))
        self.last_conv = nn.Sequential(
                                nn.Conv2d(
                                    decoder_channel_list[-1], 
                                    1,
                                    (1, 1)), 
                                nn.ReLU())
    
    def forward(self, x):
        for upblock in self.up_blocks:
            x = upblock(x)

        encoder_outs = []
        for i, layer in enumerate(self.encoder_list):
            x = layer(x)
            encoder_outs.append(x)

        encoder_outs.pop()
        encoder_len = len(encoder_outs)
        for i, layer in enumerate(self.decoder_list):
            if i == 0:
                x = self.decoder_list[0](x)
            else:
                x_concat = torch.cat([
                                x, encoder_outs[encoder_len-i]],
                                dim=1)
                x = self.decoder_list[i](x_concat)
        out = self.last_conv(x)
        return out

if __name__ == '__main__':
    model = Model().cuda()
    x = torch.rand((8, 1, 16, 1024)).cuda()
    model(x)