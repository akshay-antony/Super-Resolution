from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetConv(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 kernel_size=(3,3), 
                 padding=(1,1), 
                 stride=(1,1)):
        super(PointNetConv, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.out_features = out_features
        self.in_features = in_features
        self.weights = nn.Parameter(data=torch.Tensor(1, 1, out_features, 4), requires_grad=True)
        self.layer = nn.Conv1d(in_features, out_features, 1)
        
    def forward(self, 
                x=None, 
                X=None, 
                A=None, 
                E=None, 
                M=None):
        unfolded_x = F.unfold(x, self.kernel_size, self.padding, self.stride)
        unfolded_X = F.unfold(X, self.kernel_size, self.padding, self.stride)
        unfolded_A = F.unfold(A, self.kernel_size, self.padding, self.stride)
        unfolded_E = F.unfold(E, self.kernel_size, self.padding, self.stride)
        unfolded_M = F.unfold(M, self.kernel_size, self.padding, self.stride)

        X_c = X.view(X.shape[0], X.shape[1], -1)
        A_c = A.view(A.shape[0], A.shape[1], -1)
        E_c = E.view(E.shape[0], E.shape[1], -1)
        M_c = M.view(M.shape[0], M.shape[1], -1)

        X_c = torch.repeat_interleave(X_c, self.kernel_size[0]*self.kernel_size[1], 1)
        A_c = torch.repeat_interleave(A_c, self.kernel_size[0]*self.kernel_size[1], 1)
        E_c = torch.repeat_interleave(E_c, self.kernel_size[0]*self.kernel_size[1], 1)
        M_c = torch.repeat_interleave(M_c, self.kernel_size[0]*self.kernel_size[1], 1)
        
        x_ = unfolded_x.unsqueeze(3)           #.permute((0,2,1)).unsqueeze(3)
        X_ = (unfolded_X - X_c).unsqueeze(-1)  #.permute((0,2,1)).unsqueeze(3)
        A_ = (unfolded_A - A_c).unsqueeze(-1)  #.permute((0,2,1)).unsqueeze(3)
        E_ = (unfolded_E - E_c).unsqueeze(-1)  #.permute((0,2,1)).unsqueeze(3)

        ####
        first_feature = x_ * torch.cos(A_) * torch.cos(E_) - X_c.unsqueeze(-1)
        second_feature = x_ * torch.cos(A_) * torch.sin(E_)
        third_feature = x_ * torch.sin(A_)

        M_valid = M_c * unfolded_M
        M_valid = M_valid.unsqueeze(1)
        ####
        total_features = torch.cat([x_, 
                                    first_feature, 
                                    second_feature, 
                                    third_feature], axis=3)
                        
        total_features = total_features.permute((0, 3, 1, 2)) 
        total_features_flattened = total_features.reshape((total_features.shape[0], total_features.shape[1], -1))
        out = self.layer(total_features_flattened)
        out = out.reshape((total_features.shape[0],
                           out.shape[1],
                           total_features.shape[2],
                           total_features.shape[3]))
        out = out * M_valid
        out = torch.max(out, dim=2)[0].reshape(out.shape[0],
                                               out.shape[1],
                                               x.shape[2],
                                               x.shape[3])
        return out

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
                 dropout_rate=0.25,
                 encoder_channel_list=[64, 64*2, 64*4,
                               64*8, 64*16],
                 decoder_channel_list=[64*16, 64*8, 64*4,
                                       64*2, 64],
                 is_pointnet=False,
                 A=None,
                 E=None) -> None:
        super(Model, self).__init__()
        
        self.A = A.unsqueeze(0).unsqueeze(0)
        self.E = E.unsqueeze(0).unsqueeze(0)
        self.is_pointnet = is_pointnet
        self.up_blocks = nn.ModuleList()
        self.encoder_list = nn.ModuleList()
        self.decoder_list = nn.ModuleList()

        if self.is_pointnet:
            print("Using PointNet layer")
            self.pointnet_kernel = PointNetConv(4,
                                                out_features=32)
            in_channels = 32

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
        M = torch.where(x > 0, 1, 0).type(torch.FloatTensor).cuda()
        if self.is_pointnet:
            x = self.pointnet_kernel(x, 
                                     x, 
                                     self.A, 
                                     self.E, 
                                     M)

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