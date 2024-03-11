from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import numpy as np
import math


class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            #nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            #nn.GELU(),
        )

    def forward(self, x):

        x = self.conv(x)
        return x


class ConvResBlock(nn.Module):
    """
    标准残差快
    """
    def __init__(self, in_ch, out_ch):
        super(ConvResBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
        )
        self.conv_1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_1 = self.conv(x)
        x_res = self.conv_1(x)
        x = self.relu(x_1+x_res)
        return x


class DSConv_Block(nn.Module):
    """
    深度可分离卷积快
    """
    def __init__(self, in_ch, out_ch):
        super(DSConv_Block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=True, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, groups=out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_1 = self.conv(x)
        return x_1


class DSConvResBlock(nn.Module):
    """
    深度可分离卷积残差快
    """
    def __init__(self, in_ch, out_ch):
        super(DSConvResBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=True, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, groups=out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch),
        )
        self.conv_1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_1 = self.conv(x)
        x_res = self.conv_1(x)
        x = self.relu(x_1+x_res)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class ECA_Att(nn.Module):
    def __init__(self, channel, gamma=2,b=1):
        super(ECA_Att, self).__init__()
        t = int(abs((np.log2(channel)+b)/gamma))
        self.k = t if t % 2 else t+1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=self.k, padding=int(self.k/2), bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        x = x * y.expand_as(x)
        return x


class Denoise_Block(nn.Module):
    def __init__(self, dim=3, mid_ch=16):
        super(Denoise_Block, self).__init__()

        # 2D laplacian kernel (2D LOG operator.).
        self.channel_dim = dim
        self.mid = mid_ch
        laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

        laplacian_kernel = np.repeat(laplacian_kernel[None, None, :, :], dim, 0)
        # learnable kernel.
        self.kernel = torch.nn.Parameter(torch.FloatTensor(laplacian_kernel))
        # self.kernel = Variable(torch.FloatTensor(laplacian_kernel).to(device))
        # self.Conv = nn.Sequential(
        #     nn.Conv2d(3, 3, 3, 1, 1),  # nn.Conv2d(3, self.mid, 7, 1, 3),
        #     nn.BatchNorm2d(3),
        #     nn.ReLU(),
        # )

        self.det_conv = nn.Sequential(
            nn.Conv2d(3, self.mid, 3, 1, 1),  # nn.Conv2d(3, self.mid, 7, 1, 3),
            nn.BatchNorm2d(self.mid),
            nn.ReLU(),
            nn.Conv2d(self.mid, self.mid, 7, 1, 3),
            nn.BatchNorm2d(self.mid),
            nn.ReLU(),
            nn.Conv2d(self.mid, self.mid, 7, 1, 3),
            nn.BatchNorm2d(self.mid),
            nn.ReLU(),
            #ECA_Att(self.mid),
            nn.Conv2d(self.mid, 1, kernel_size=1, stride=1, padding=0),
            #nn.BatchNorm2d(self.mid),
            #nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        lap = F.conv2d(x, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        # lap = self.Conv(lap)
        noise_map = self.det_conv(x+lap)
        out = x * (1-noise_map)  # (1-noise_map)  #####改动
        return out, lap, noise_map


class DDCA_UNet(nn.Module):
    """
    改进方法
    """
    def __init__(self, img_ch=3, output_ch=2, feature_show=False):
        super(DDCA_UNet, self).__init__()

        self.show = feature_show

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.n_classes = output_ch

        self.de_noise = Denoise_Block()

        self.ECA1 = ECA_Att(channel=filters[0])
        self.ECA2 = ECA_Att(channel=filters[1])
        self.ECA3 = ECA_Att(channel=filters[2])
        self.ECA4 = ECA_Att(channel=filters[3])
        # self.ECA5 = ECA_Att(channel=filters[3])

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Down4 = nn.Conv2d(filters[3], filters[3], kernel_size=2, stride=2, padding=0, groups=filters[3])

        self.Conv1 = ConvResBlock(img_ch*2, filters[0])
        self.Conv2 = DSConv_Block(filters[0], filters[1])
        self.Conv3 = DSConv_Block(filters[1], filters[2])
        self.Conv4 = DSConv_Block(filters[2], filters[3])
        self.Conv5 = DSConv_Block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = DSConv_Block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = DSConv_Block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = DSConv_Block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = ConvResBlock(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        x, lap, noise_map = self.de_noise(x)

        e1 = self.Conv1(torch.cat((x, lap),dim=1))    # (torch.cat((x, lap),dim=1))
        e1 = self.ECA1(e1)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e2 = self.ECA2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e3 = self.ECA3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        e4 = self.ECA4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #  out = self.active(out)
        if self.show:
            return out, lap, noise_map, x
        else:
            return out


class abalation_Denoise_Block(nn.Module):
    def __init__(self, dim=3, mid_ch=16):
        super(abalation_Denoise_Block, self).__init__()

        # 2D laplacian kernel (2D LOG operator.).
        self.channel_dim = dim
        self.mid = mid_ch
        laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

        laplacian_kernel = np.repeat(laplacian_kernel[None, None, :, :], dim, 0)
        # learnable kernel.
        self.kernel = torch.nn.Parameter(torch.FloatTensor(laplacian_kernel))
        # self.kernel = Variable(torch.FloatTensor(laplacian_kernel).to(device))
        # self.Conv = nn.Sequential(
        #     nn.Conv2d(3, 3, 3, 1, 1),  # nn.Conv2d(3, self.mid, 7, 1, 3),
        #     nn.BatchNorm2d(3),
        #     nn.ReLU(),
        # )

        self.det_conv = nn.Sequential(
            nn.Conv2d(3, self.mid, 3, 1, 1),  # nn.Conv2d(3, self.mid, 7, 1, 3),
            nn.BatchNorm2d(self.mid),
            nn.ReLU(),
            nn.Conv2d(self.mid, self.mid, 7, 1, 3),
            nn.BatchNorm2d(self.mid),
            nn.ReLU(),
            nn.Conv2d(self.mid, self.mid, 7, 1, 3),
            nn.BatchNorm2d(self.mid),
            nn.ReLU(),
            #ECA_Att(self.mid),
            nn.Conv2d(self.mid, 1, kernel_size=1, stride=1, padding=0),
            #nn.BatchNorm2d(self.mid),
            #nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        # lap = F.conv2d(x, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        # lap = self.Conv(lap)
        noise_map = self.det_conv(x)
        out = x * (1-noise_map)  # (1-noise_map)  #####改动
        return out, noise_map


class NoLap_DDCA_UNet(nn.Module):
    """
    改进方法
    """
    def __init__(self, img_ch=3, output_ch=2, feature_show=False):
        super(NoLap_DDCA_UNet, self).__init__()

        self.show = feature_show

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.n_classes = output_ch

        self.de_noise = abalation_Denoise_Block()

        self.ECA1 = ECA_Att(channel=filters[0])
        self.ECA2 = ECA_Att(channel=filters[1])
        self.ECA3 = ECA_Att(channel=filters[2])
        self.ECA4 = ECA_Att(channel=filters[3])
        # self.ECA5 = ECA_Att(channel=filters[3])

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Down4 = nn.Conv2d(filters[3], filters[3], kernel_size=2, stride=2, padding=0, groups=filters[3])

        self.Conv1 = ConvResBlock(img_ch, filters[0])
        self.Conv2 = DSConv_Block(filters[0], filters[1])
        self.Conv3 = DSConv_Block(filters[1], filters[2])
        self.Conv4 = DSConv_Block(filters[2], filters[3])
        self.Conv5 = DSConv_Block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = DSConv_Block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = DSConv_Block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = DSConv_Block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = ConvResBlock(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        x, noise_map = self.de_noise(x)

        e1 = self.Conv1(x)    # (torch.cat((x, lap),dim=1))
        e1 = self.ECA1(e1)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e2 = self.ECA2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e3 = self.ECA3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        e4 = self.ECA4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #  out = self.active(out)
        if self.show:
            return out, noise_map
        else:
            return out


class DU_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=1):
        super(DU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.de_noise = Denoise_Block()

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch*2, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        x, lap, noise_map = self.de_noise(x)

        e1 = self.Conv1(torch.cat((x, lap), dim=1))  # (torch.cat((x, lap),dim=1))

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out


class ECA_UNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=1):
        super(ECA_UNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.ECA1 = ECA_Att(channel=filters[0])
        self.ECA2 = ECA_Att(channel=filters[1])
        self.ECA3 = ECA_Att(channel=filters[2])
        self.ECA4 = ECA_Att(channel=filters[3])

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)
        e1 = self.ECA1(e1)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e2 = self.ECA2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e3 = self.ECA3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        e4 = self.ECA4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out


class add_Block(nn.Module):
    def __init__(self, dim=3, mid_ch=16):
        super(add_Block, self).__init__()

        # 2D laplacian kernel (2D LOG operator.).
        self.channel_dim = dim
        self.mid = mid_ch

        self.det_conv = nn.Sequential(
            nn.Conv2d(3, 3, 3, 1, 1,groups=self.channel_dim),  # nn.Conv2d(3, self.mid, 7, 1, 3),
            nn.Conv2d(3, self.mid, 3, 1, 1),  # nn.Conv2d(3, self.mid, 7, 1, 3),
            nn.BatchNorm2d(self.mid),
            nn.ReLU(),
            nn.Conv2d(self.mid, self.mid, 7, 1, 3),
            nn.BatchNorm2d(self.mid),
            nn.ReLU(),
            nn.Conv2d(self.mid, self.mid, 7, 1, 3),
            nn.BatchNorm2d(self.mid),
            nn.ReLU(),
            #ECA_Att(self.mid),
            nn.Conv2d(self.mid, 3, kernel_size=1, stride=1, padding=0),
            #nn.BatchNorm2d(self.mid),
            #nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        # lap = self.Conv(lap)
        out = self.det_conv(x)
        return out


class add_UNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=1):
        super(add_UNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.faked = add_Block()

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.faked(x)
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #d1 = self.active(out)

        return out




